"""Evaluation loop: run an inference method on a dataset and report metrics."""

from __future__ import annotations

import time
import logging
from typing import Any

from tqdm import tqdm

from src.methods.base import BaseMethod
from src.utils.helpers import (
    LATENTMAS_CODE_TASK_SOURCES,
    MINERVA_MATH_SOURCES,
    extract_latentmas_eval_prediction,
    extract_markdown_python_block,
    latentmas_eval_is_correct,
    normalize_answer,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """Run an inference method on a dataset and report accuracy / timing / tokens.

    Per-task scoring is delegated to ``latentmas_eval_is_correct`` in
    ``src.utils.helpers`` (the helper name is internal; see the README's
    *Evaluation protocol* section for the per-dataset rules).
    """

    DEFAULT_REGEX = r"####\s*([\-\d\.\,]+)"

    def __init__(self, config: dict):
        self.config = config
        self.eval_config = config.get("evaluation", {})
        self.answer_regex = self.eval_config.get(
            "extract_answer_regex", self.DEFAULT_REGEX
        )
        self.regex_by_source = self.eval_config.get(
            "extract_answer_regex_by_dataset", {}
        )
        self.generate_bs = max(1, self.eval_config.get("generate_bs", 1))

    def _get_pred_answer_for_sample(self, source: str, pred_text: str) -> str:
        """Short string for JSON logging (same extraction as scoring)."""
        if source in LATENTMAS_CODE_TASK_SOURCES:
            block = extract_markdown_python_block(pred_text)
            return block[:200] + "..." if block and len(block) > 200 else (block or "")
        raw = extract_latentmas_eval_prediction(
            pred_text,
            source,
            extract_answer_regex=self.answer_regex,
            extract_answer_regex_by_dataset=self.regex_by_source,
        )
        if not raw:
            return pred_text.strip()[:100]
        return normalize_answer(raw) if raw else pred_text.strip()[:100]

    def _is_correct_for_sample(
        self, source: str, pred_text: str, gold_answer: str
    ) -> bool:
        return latentmas_eval_is_correct(
            pred_text,
            gold_answer,
            source,
            extract_answer_regex=self.answer_regex,
            extract_answer_regex_by_dataset=self.regex_by_source,
        )

    def evaluate(
        self,
        method: BaseMethod,
        dataset: list[dict],
    ) -> dict[str, Any]:
        """Run evaluation.

        Args:
            method: initialized method (setup already called).
            dataset: list of {"question", "answer", "source"}.

        Returns:
            Dict with metrics: accuracy, avg_time, per_sample results.
        """
        results: list[dict[str, Any]] = []
        correct = 0
        total = 0
        total_time = 0.0

        # solve_batch may internally loop over solve() for some methods, so a
        # large generate_bs can appear "stuck"; update the progress bar per
        # sample once outputs become available.
        pbar = tqdm(dataset, desc="Evaluating")
        idx = 0
        while idx < len(dataset):
            batch = dataset[idx : idx + self.generate_bs]
            batch_questions = [s["question"] for s in batch]

            t0 = time.time()
            outputs = method.solve_batch(batch_questions)
            elapsed = time.time() - t0
            total_time += elapsed

            if len(outputs) != len(batch):
                raise RuntimeError(
                    f"solve_batch returned {len(outputs)} results for {len(batch)} samples"
                )

            per_time = elapsed / len(batch) if batch else 0.0
            for sample, output in zip(batch, outputs):
                question = sample["question"]
                gold_answer = sample["answer"]
                pred_text = output["answer"]
                source = sample.get("source", "")
                truncated = bool(
                    output.get("metadata", {}).get(
                        "truncated_by_max_new_tokens", False
                    )
                )
                sample_token_usage = output.get("metadata", {}).get(
                    "token_usage", {}
                )

                is_correct = self._is_correct_for_sample(
                    source, pred_text, gold_answer
                )
                pred_answer = self._get_pred_answer_for_sample(source, pred_text)
                if source in MINERVA_MATH_SOURCES:
                    gold_normalized = gold_answer
                else:
                    gold_normalized = normalize_answer(gold_answer)
                if is_correct:
                    correct += 1
                total += 1

                tqdm.write(
                    f"[sample {total}] correct={is_correct} "
                    f"truncated_by_max_new_tokens={truncated} "
                    f"tokens={sample_token_usage.get('total_tokens', 'N/A')}"
                )

                results.append(
                    {
                        "question": question,
                        "gold_answer": gold_normalized,
                        "predicted_answer": pred_answer or "",
                        "correct": is_correct,
                        "wall_time": per_time,
                        "full_output": pred_text,
                        "truncated_by_max_new_tokens": truncated,
                        "metadata": output.get("metadata", {}),
                    }
                )
                pbar.update(1)
                pbar.set_postfix(
                    acc=f"{correct / total:.2%}",
                    time=f"{per_time:.1f}s",
                    tokens=sample_token_usage.get("total_tokens", "N/A"),
                )

            idx += len(batch)

        accuracy = correct / total if total > 0 else 0.0
        avg_time = total_time / total if total > 0 else 0.0
        speed_sps = (total / total_time) if total_time > 0 else 0.0

        all_prompt_tokens = [
            r["metadata"].get("token_usage", {}).get("prompt_tokens", 0)
            for r in results
        ]
        all_completion_tokens = [
            r["metadata"].get("token_usage", {}).get("completion_tokens", 0)
            for r in results
        ]
        all_total_tokens = [
            r["metadata"].get("token_usage", {}).get("total_tokens", 0)
            for r in results
        ]
        sum_prompt_tokens = sum(all_prompt_tokens)
        sum_completion_tokens = sum(all_completion_tokens)
        sum_total_tokens = sum(all_total_tokens)
        avg_prompt_tokens = sum_prompt_tokens / total if total > 0 else 0.0
        avg_completion_tokens = sum_completion_tokens / total if total > 0 else 0.0
        avg_total_tokens = sum_total_tokens / total if total > 0 else 0.0

        token_usage_stats = {
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_total_tokens": avg_total_tokens,
            "sum_prompt_tokens": sum_prompt_tokens,
            "sum_completion_tokens": sum_completion_tokens,
            "sum_total_tokens": sum_total_tokens,
        }

        n_truncated = sum(1 for r in results if r["truncated_by_max_new_tokens"])
        incorrect = total - correct
        trunc_in_correct = sum(
            1 for r in results if r["correct"] and r["truncated_by_max_new_tokens"]
        )
        trunc_in_incorrect = sum(
            1
            for r in results
            if not r["correct"] and r["truncated_by_max_new_tokens"]
        )
        rate_trunc_correct = (
            trunc_in_correct / correct if correct > 0 else None
        )
        rate_trunc_incorrect = (
            trunc_in_incorrect / incorrect if incorrect > 0 else None
        )

        truncation_stats = {
            "total_truncated": n_truncated,
            "truncated_among_correct": trunc_in_correct,
            "truncated_among_incorrect": trunc_in_incorrect,
            "rate_truncated_in_correct": rate_trunc_correct,
            "rate_truncated_in_incorrect": rate_trunc_incorrect,
        }

        summary = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "avg_time": avg_time,
            "total_time": total_time,
            "speed_sps": speed_sps,
            "method": self.config.get("method", "unknown"),
            "dataset": self.config.get("data", {}).get("dataset", "unknown"),
            "truncation_stats": truncation_stats,
            "token_usage_stats": token_usage_stats,
            "results": results,
        }

        logger.info(
            f"[Evaluation] {summary['method']} on {summary['dataset']}: "
            f"accuracy={accuracy:.2%} ({correct}/{total}), "
            f"avg_time={avg_time:.2f}s"
        )

        eval_summary_line = (
            f"[Eval] accuracy: {accuracy:.2%}  |  "
            f"speed: {speed_sps:.2f} samples/s  (avg {avg_time:.2f} s/sample)"
        )
        if correct > 0 and rate_trunc_correct is not None:
            trunc_summary = (
                f"[Eval] truncated by max_new_tokens: {n_truncated}/{total} | "
                f"truncation rate among correct: {rate_trunc_correct:.2%} "
                f"({trunc_in_correct}/{correct})"
            )
        else:
            trunc_summary = (
                f"[Eval] truncated by max_new_tokens: {n_truncated}/{total} | "
                "truncation rate among correct: N/A (no correct samples)"
            )
        if incorrect > 0 and rate_trunc_incorrect is not None:
            trunc_summary_wrong = (
                f"truncation rate among incorrect: {rate_trunc_incorrect:.2%} "
                f"({trunc_in_incorrect}/{incorrect})"
            )
        else:
            trunc_summary_wrong = (
                "truncation rate among incorrect: N/A (no incorrect samples)"
            )
        token_summary = (
            f"[Eval] tokens: "
            f"avg_prompt={avg_prompt_tokens:.1f}, "
            f"avg_completion={avg_completion_tokens:.1f}, "
            f"avg_total={avg_total_tokens:.1f} | "
            f"sum prompt={sum_prompt_tokens}, "
            f"completion={sum_completion_tokens}, "
            f"total={sum_total_tokens}"
        )
        print(
            f"\n{eval_summary_line}\n{trunc_summary}\n{trunc_summary_wrong}"
            f"\n{token_summary}\n"
        )
        logger.info(eval_summary_line)
        logger.info(trunc_summary + " | " + trunc_summary_wrong)
        logger.info(token_summary)

        return summary
