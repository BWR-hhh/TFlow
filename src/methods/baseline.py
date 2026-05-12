"""Baseline: single-agent inference (no inter-agent communication)."""

from __future__ import annotations

import time
import logging
from typing import Any

from src.methods.base import BaseMethod
from src.models.agent import LLMAgent
from src.utils.helpers import get_device, resolve_system_prompt_for_dataset

logger = logging.getLogger(__name__)


class BaselineMethod(BaseMethod):
    """Single LLM agent, no multi-agent communication."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.agent: LLMAgent | None = None

    def setup(self) -> None:
        dataset = self.config.get("data", {}).get("dataset", "")
        system_prompt = resolve_system_prompt_for_dataset(
            self.method_config, dataset
        )
        self.agent = LLMAgent(
            model_name=self.model_config["name"],
            agent_id=0,
            role="solver",
            system_prompt=system_prompt,
            dtype=self.model_config.get("dtype", "bfloat16"),
            device=get_device(),
            attn_implementation=self.model_config.get("attn_implementation"),
            device_map=self.model_config.get("device_map"),
            max_memory=self.model_config.get("max_memory"),
        )
        self.generate_bs = max(
            1,
            self.config.get("evaluation", {}).get("generate_bs", 1),
        )
        logger.info("[Baseline] Setup complete (generate_bs=%d).", self.generate_bs)

    def solve(self, question: str) -> dict[str, Any]:
        t0 = time.time()
        response, truncated, token_usage = self.agent.generate_with_truncation_info(
            question,
            max_new_tokens=self.model_config.get("max_new_tokens", 1024),
            temperature=self.model_config.get("temperature", 0.7),
            top_p=self.model_config.get("top_p", 0.95),
            do_sample=self.model_config.get("do_sample", True),
        )
        elapsed = time.time() - t0

        return {
            "answer": response,
            "reasoning": response,
            "metadata": {
                "method": "baseline",
                "num_agents": 1,
                "wall_time": elapsed,
                "truncated_by_max_new_tokens": truncated,
                "token_usage": token_usage,
            },
        }

    def solve_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        """Truly batched generation via ``generate_batch_with_truncation_info``."""
        if not questions:
            return []
        gen_kwargs = dict(
            max_new_tokens=self.model_config.get("max_new_tokens", 1024),
            temperature=self.model_config.get("temperature", 0.7),
            top_p=self.model_config.get("top_p", 0.95),
            do_sample=self.model_config.get("do_sample", True),
        )
        t0_batch = time.time()
        b = len(questions)
        triples = self.agent.generate_batch_with_truncation_info(
            questions, **gen_kwargs
        )
        elapsed = time.time() - t0_batch
        batch_mean = elapsed / b if b else 0.0
        return [
            {
                "answer": resp,
                "reasoning": resp,
                "metadata": {
                    "method": "baseline",
                    "num_agents": 1,
                    "wall_time": batch_mean,
                    "batch_mean_wall_time": batch_mean,
                    "batch_size": b,
                    "batched": True,
                    "truncated_by_max_new_tokens": truncated,
                    "token_usage": token_usage,
                },
            }
            for resp, truncated, token_usage in triples
        ]