"""Benchmark dataset loaders.

Five evaluation datasets are supported:

* ``gsm8k``         — grade-school math word problems
  (``openai/gsm8k`` ``main`` ``test``).
* ``mbppplus``      (alias ``mbpp``)      — MBPP+ Python coding
  (``evalplus/mbppplus``).
* ``humanevalplus`` (alias ``humaneval``) — HumanEval+ Python coding
  (``evalplus/humanevalplus``).
* ``minerva_math``  — Minerva Math with the 4-shot prompt from
  ``EleutherAI/lm-evaluation-harness`` (``math-ai/minervamath``;
  ``knoveleng/Minerva-Math`` is used as a fallback).
* ``mmlu``          — MMLU-Redux 2.0 (``edinburgh-dawg/mmlu-redux-2.0``;
  ``cais/mmlu`` and ``hails/mmlu_no_train`` are used as fallbacks).
"""

from __future__ import annotations

import os
import re
import logging
from typing import Optional

from datasets import load_dataset

from src.utils.helpers import (
    last_boxed_only_string,
    normalize_final_answer,
    remove_boxed,
)

logger = logging.getLogger(__name__)


_MINERVA_MATH_IDS = ["math-ai/minervamath", "knoveleng/Minerva-Math"]
_MMLU_REDUX_ID = "edinburgh-dawg/mmlu-redux-2.0"
_MMLU_FALLBACK_IDS = ["cais/mmlu", "hails/mmlu_no_train"]
_MMLU_CHOICE_LABELS = ("A", "B", "C", "D")

# Canonical dataset key → user-facing aliases (CLI / YAML)
_DATASET_ALIASES: dict[str, list[str]] = {
    "gsm8k": ["gsm8k"],
    "mbppplus": ["mbppplus", "mbpp_plus", "mbpp", "mbpp-plus"],
    "humanevalplus": [
        "humanevalplus",
        "humaneval_plus",
        "humaneval",
        "humaneval-plus",
    ],
    "minerva_math": ["minerva_math", "minervamath", "minerva-math"],
    "mmlu": [
        "mmlu",
        "mmlu_redux",
        "mmlu-redux",
        "mmlu_redux_2.0",
        "mmlu-redux-2.0",
        "MMLU",
    ],
}


def resolve_dataset_name(raw: str) -> str:
    """Map a user-facing name (with dashes/alt spelling) to the canonical key."""
    key = raw.strip().lower().replace("-", "_")
    for canonical, aliases in _DATASET_ALIASES.items():
        if key in aliases:
            return canonical
    return key


def effective_hf_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """Resolve the Hugging Face ``datasets`` cache directory.

    Returns ``data.cache_dir`` (with ``~`` expansion) when set, otherwise
    ``None`` so that ``datasets`` falls back to ``HF_DATASETS_CACHE`` /
    ``HF_HOME`` / its built-in default (``~/.cache/huggingface``).
    """
    if cache_dir:
        return os.path.expanduser(str(cache_dir).strip())
    return None


def _normalize_answer(ans: Optional[str]) -> str:
    if ans is None:
        return ""
    return str(ans).strip().lower()


# ---------------------------------------------------------------------------
# GSM8K
# ---------------------------------------------------------------------------

def _load_gsm8k(split: str, cache_dir: Optional[str] = None) -> list[dict]:
    ds = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    results: list[dict] = []
    for item in ds:
        question = item["question"].strip()
        answer_text = item["answer"]
        match = re.search(r"####\s*([\-\d\.\,]+)", answer_text)
        answer = match.group(1).replace(",", "").strip() if match else answer_text
        results.append(
            {
                "question": question,
                "answer": _normalize_answer(answer),
                "source": "gsm8k",
            }
        )
    return results


# ---------------------------------------------------------------------------
# MBPP+
# ---------------------------------------------------------------------------

def _load_mbppplus(split: str = "test", cache_dir: Optional[str] = None) -> list[dict]:
    ds = load_dataset("evalplus/mbppplus", None, split=split, cache_dir=cache_dir)
    results: list[dict] = []
    for item in ds:
        prompt = item["prompt"]
        test_list = item["test_list"]
        test_str = str(item["test"])
        question = (
            "Please provide a self-contained Python script that solves the following "
            "problem in a markdown code block:\n"
            "```python\nYOUR_PYTHON_CODE\n```:\n"
            f"{prompt}\nYour answer will be tested on test cases like:\n"
            + "\n".join(test_list[:3])
        )
        results.append(
            {"question": question, "answer": test_str, "source": "mbppplus"}
        )
    return results


# ---------------------------------------------------------------------------
# HumanEval+
# ---------------------------------------------------------------------------

def _load_humanevalplus(split: str = "test", cache_dir: Optional[str] = None) -> list[dict]:
    ds = load_dataset("evalplus/humanevalplus", None, split=split, cache_dir=cache_dir)
    results: list[dict] = []
    for item in ds:
        prompt = item["prompt"]
        entry_point = item["entry_point"]
        raw_test = str(item["test"])
        answer = raw_test.replace("candidate", entry_point) + f"\n\ncheck({entry_point})"
        question = (
            "Please provide a self-contained Python script that solves the following "
            "problem in a markdown code block:\n"
            "```python\nYOUR_PYTHON_CODE\n```:\n"
            f"{prompt}"
        )
        results.append(
            {"question": question, "answer": answer, "source": "humanevalplus"}
        )
    return results


# ---------------------------------------------------------------------------
# Minerva Math (with 4-shot prompt aligned with lm-evaluation-harness)
# ---------------------------------------------------------------------------

# Standard 4-shot exemplars for Minerva Math (aligned with
# EleutherAI/lm-evaluation-harness → tasks/minerva_math/utils.py:list_fewshot_samples,
# which reproduces Appendix D of Lewkowycz et al. 2022).
_MINERVA_MATH_FEWSHOT: list[tuple[str, str]] = [
    (
        r"Find the domain of the expression $\frac{\sqrt{x-2}}{\sqrt{5-x}}$.",
        r"The expressions inside each square root must be non-negative. "
        r"Therefore, $x-2 \ge 0$, so $x \ge 2$, and $5 - x \ge 0$, so $x \le 5$. "
        r"Also, the denominator cannot be equal to zero, so $5-x>0$, which gives $x<5$. "
        r"Therefore, the domain of the expression is $\boxed{[2,5)}$."
        "\nFinal Answer: The final answer is $[2,5)$. I hope it is correct.",
    ),
    (
        r"If $\det \mathbf{A} = 2$ and $\det \mathbf{B} = 12,$ then find $\det (\mathbf{A} \mathbf{B}).$",
        r"We have that $\det (\mathbf{A} \mathbf{B}) = (\det \mathbf{A})(\det \mathbf{B}) = (2)(12) = \boxed{24}.$"
        "\nFinal Answer: The final answer is $24$. I hope it is correct.",
    ),
    (
        "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, "
        "how many times must Terrell lift them in order to lift the same total weight?",
        r"If Terrell lifts two 20-pound weights 12 times, he lifts a total of "
        r"$2 \cdot 12 \cdot 20 = 480$ pounds of weight. If he lifts two 15-pound weights "
        r"instead for $n$ times, he will lift a total of $2 \cdot 15 \cdot n = 30n$ pounds of weight. "
        r"Equating this to 480 pounds, we can solve for $n$:"
        "\n\\begin{align*}\n30n &= 480 \\\\\n\\Rightarrow\\qquad n &= 480/30 = \\boxed{16}\n\\end{align*}"
        "\nFinal Answer: The final answer is $16$. I hope it is correct.",
    ),
    (
        "If the system of equations\n\n\\begin{align*}\n6x-4y &= a, \\\\\n6y-9x &= b.\n\\end{align*}"
        "has a solution $(x, y)$ where $x$ and $y$ are both nonzero, "
        r"find $\frac{a}{b},$ assuming $b$ is nonzero.",
        r"If we multiply the first equation by $-\frac{3}{2}$, we obtain"
        "\n\n"
        r"$$6y - 9x = -\frac{3}{2}a.$$"
        r"Since we also know that $6y - 9x = b$, we have"
        "\n\n"
        r"$$-\frac{3}{2}a = b \Rightarrow \frac{a}{b} = \boxed{-\frac{2}{3}}.$$"
        "\nFinal Answer: The final answer is $-\\frac{2}{3}$. I hope it is correct.",
    ),
]


def _build_minerva_4shot_prefix() -> str:
    parts = []
    for problem, solution in _MINERVA_MATH_FEWSHOT:
        parts.append(f"Problem:\n{problem}\n\nSolution:\n{solution}")
    return "\n\n".join(parts)


def _load_minerva_math(
    split: str = "train",
    cache_dir: Optional[str] = None,
    loader_extra: Optional[dict] = None,
) -> list[dict]:
    """Minerva Math (~272 STEM problems from MIT OCW).

    By default wraps each query in a 4-shot Minerva prompt.  Override via
    ``loader_extra.minerva_math.n_shot: 0`` to disable few-shot wrapping.
    """
    extra = loader_extra or {}
    n_shot = int(extra.get("n_shot", 4))
    if n_shot not in (0, 4):
        raise ValueError(f"minerva_math: n_shot={n_shot} not supported (only 0 or 4).")

    ds = None
    for repo_id in _MINERVA_MATH_IDS:
        try:
            ds = load_dataset(
                repo_id, split=split, cache_dir=cache_dir, trust_remote_code=True
            )
            break
        except Exception:
            continue
    if ds is None:
        raise RuntimeError(
            f"Cannot load Minerva Math from any of {_MINERVA_MATH_IDS}. "
            "Check network or HF_TOKEN."
        )

    prefix = _build_minerva_4shot_prefix() if n_shot == 4 else ""
    results: list[dict] = []
    for item in ds:
        problem = str(item.get("problem", item.get("question", ""))).strip()
        solution = str(item.get("solution", item.get("answer", ""))).strip()

        boxed = last_boxed_only_string(solution)
        if boxed is not None:
            answer = normalize_final_answer(remove_boxed(boxed))
        else:
            answer = normalize_final_answer(solution)

        if n_shot == 4:
            question = f"{prefix}\n\nProblem:\n{problem}\n\nSolution:\n"
        else:
            question = problem

        results.append(
            {"question": question, "answer": answer, "source": "minerva_math"}
        )
    return results


# ---------------------------------------------------------------------------
# MMLU (Redux 2.0 with original MMLU fallback)
# ---------------------------------------------------------------------------

def _resolve_mmlu_corrected_answer(corr: str, choices: list) -> Optional[str]:
    """Map a Redux ``correct_answer`` value back to an A-D letter."""
    c = corr.strip()
    if len(c) == 1 and c.upper() in _MMLU_CHOICE_LABELS:
        return c.upper()
    try:
        idx = int(c)
        if 0 <= idx < len(_MMLU_CHOICE_LABELS):
            return _MMLU_CHOICE_LABELS[idx]
    except ValueError:
        pass
    c_lower = c.lower()
    for i, opt in enumerate(choices):
        if i >= len(_MMLU_CHOICE_LABELS):
            break
        if str(opt).strip().lower() == c_lower:
            return _MMLU_CHOICE_LABELS[i]
    return None


def _load_mmlu(split: str = "test", cache_dir: Optional[str] = None) -> list[dict]:
    """MMLU-Redux 2.0 (5,700 manually re-annotated questions across 57 subjects).

    Filtering:
    * ``error_type == "no_correct_answer"`` → drop.
    * ``error_type == "wrong_groundtruth"`` → use corrected answer.
    * Falls back to original MMLU (``cais/mmlu`` / ``hails/mmlu_no_train``).
    """
    ds = None
    is_redux = False

    try:
        from datasets import get_dataset_config_names
        configs = get_dataset_config_names(_MMLU_REDUX_ID)
        parts = []
        for cfg in configs:
            try:
                part = load_dataset(
                    _MMLU_REDUX_ID, cfg, split=split,
                    cache_dir=cache_dir, trust_remote_code=True,
                )
                parts.append(part)
            except Exception:
                continue
        if parts:
            from datasets import concatenate_datasets
            ds = concatenate_datasets(parts)
            is_redux = True
            logger.info(
                "MMLU-Redux 2.0: loaded %d samples from %d/%d subject configs.",
                len(ds), len(parts), len(configs),
            )
    except Exception as e:
        logger.info("MMLU-Redux 2.0 unavailable (%s), trying fallback.", e)

    if ds is None:
        for repo_id in _MMLU_FALLBACK_IDS:
            try:
                ds = load_dataset(
                    repo_id, "all", split=split,
                    cache_dir=cache_dir, trust_remote_code=True,
                )
                break
            except Exception:
                continue
    if ds is None:
        raise RuntimeError(
            f"Cannot load MMLU from {_MMLU_REDUX_ID} or fallbacks {_MMLU_FALLBACK_IDS}. "
            "Check network or HF_TOKEN."
        )

    results: list[dict] = []
    skipped = 0
    corrected = 0
    for item in ds:
        stem = str(item.get("question", "")).strip()
        choices = item.get("choices", [])
        answer_idx = item.get("answer", 0)

        if not stem or len(choices) < 2:
            skipped += 1
            continue

        if is_redux:
            error_type = str(item.get("error_type", "ok")).strip().lower()
            if error_type == "no_correct_answer":
                skipped += 1
                continue

        options_text = "\n".join(
            f"{_MMLU_CHOICE_LABELS[i]}: {str(c).strip()}"
            for i, c in enumerate(choices)
            if i < len(_MMLU_CHOICE_LABELS)
        )
        question = f"{stem}\n{options_text}"

        answer = None
        if is_redux:
            error_type = str(item.get("error_type", "ok")).strip().lower()
            corr = str(item.get("correct_answer", "") or "").strip()
            if error_type == "wrong_groundtruth" and corr:
                letter = _resolve_mmlu_corrected_answer(corr, choices)
                if letter is not None:
                    answer = letter.lower()
                    corrected += 1

        if answer is None:
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(_MMLU_CHOICE_LABELS):
                answer = _MMLU_CHOICE_LABELS[answer_idx].lower()
            else:
                answer = str(answer_idx).strip().lower()

        results.append(
            {"question": question, "answer": _normalize_answer(answer), "source": "mmlu"}
        )

    if skipped:
        logger.info("MMLU: skipped %d samples (empty or no_correct_answer).", skipped)
    if corrected:
        logger.info("MMLU-Redux: corrected %d wrong_groundtruth answers.", corrected)
    return results


# ---------------------------------------------------------------------------
# Public dispatcher
# ---------------------------------------------------------------------------

_LOADERS = {
    "gsm8k": _load_gsm8k,
    "mbppplus": _load_mbppplus,
    "humanevalplus": _load_humanevalplus,
    "minerva_math": _load_minerva_math,
    "mmlu": _load_mmlu,
}


def load_dataset_for_eval(
    name: str,
    split: str = "test",
    max_samples: int = -1,
    cache_dir: Optional[str] = None,
    loader_extra: Optional[dict] = None,
) -> list[dict]:
    """Load and format an evaluation dataset.

    Returns list of dicts with keys ``question`` / ``answer`` / ``source``.
    """
    name = resolve_dataset_name(name)
    extra_for = (loader_extra or {}).get(name) if loader_extra else None
    if extra_for is not None and not isinstance(extra_for, dict):
        raise TypeError(f"loader_extra[{name!r}] must be a dict, got {type(extra_for)}")

    if name not in _LOADERS:
        raise ValueError(
            f"Unknown dataset: {name}. Choose from {list(_LOADERS.keys())} "
            "(aliases: mbpp → mbppplus, humaneval → humanevalplus)."
        )

    use_train_only = name == "minerva_math" and split == "test"
    use_split = "train" if use_train_only else split

    if name == "minerva_math":
        data = _LOADERS[name](split=use_split, cache_dir=cache_dir, loader_extra=extra_for)
    else:
        data = _LOADERS[name](split=use_split, cache_dir=cache_dir)

    if 0 < max_samples < len(data):
        data = data[:max_samples]

    logger.info(f"Loaded {len(data)} samples from {name}/{split}")
    return data
