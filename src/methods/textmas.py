"""TextMAS: text-space multi-agent inference baseline."""

from __future__ import annotations

import time
import logging
from typing import Any

from src.methods.base import BaseMethod
from src.models.agent import LLMAgent
from src.utils.helpers import get_device

logger = logging.getLogger(__name__)

_AGENT_C_CTX = (
    "The context preceding the restated task contains Agent A's strategy "
    "analysis and Agent B's knowledge extraction; integrate both into your "
    "solution."
)

# Sender agents (Agent A / Agent B) always emit at most this many tokens,
# independent of the dataset-specific ``model.max_new_tokens``.  This keeps
# the context handed to the executor bounded regardless of the executor's
# own budget.
_SENDER_MAX_NEW_TOKENS = 1024


class TextMASMethod(BaseMethod):
    """Three-agent pipeline that exchanges natural-language text.

    Stages (one round):
      1. **Agent A** and **Agent B** run in parallel on the same ``question``
         with empty context.  Their system prompts come from the YAML
         (``method_config.roles_by_dataset[<dataset>]`` if the dataset key
         matches, otherwise ``method_config.roles``) and are dataset-specific.
      2. **Agent C** (executor) runs with ``context`` set to the concatenated
         outputs of A and B (see :meth:`_build_context`), followed by the
         original ``question`` as the final user turn.

    ``num_rounds`` controls how many executor passes to run (default 1);
    each pass appends to the log so later passes may refine on the previous
    executor output.  Sender-side max-new-tokens are pinned to
    :data:`_SENDER_MAX_NEW_TOKENS`, while the executor uses the dataset's
    full ``model.max_new_tokens`` budget.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.agents: list[LLMAgent] = []

    def setup(self) -> None:
        device = get_device()
        roles_by_ds = self.method_config.get("roles_by_dataset") or {}
        dataset = self.config.get("data", {}).get("dataset", "")
        roles: list = []
        if dataset:
            from src.data.datasets import resolve_dataset_name

            key = resolve_dataset_name(dataset.split(",")[0].strip())
            if key and key in roles_by_ds:
                roles = roles_by_ds[key]
        if not roles:
            roles = self.method_config.get("roles", [])

        if not roles:
            roles = [
                {
                    "name": "Agent A",
                    "system_prompt": (
                        "You are Agent A (Strategist). Analyze HOW to approach this "
                        "task: identify the type of reasoning required (deductive, "
                        "inductive, abductive, analogical), plan the optimal "
                        "chain-of-thought structure and depth, and anticipate common "
                        "pitfalls and failure modes. Focus on the meta-cognitive "
                        "strategy, not the answer itself."
                    ),
                },
                {
                    "name": "Agent B",
                    "system_prompt": (
                        "You are Agent B (Knowledge Extractor). Identify WHAT "
                        "knowledge is needed: the relevant domain(s) and specialized "
                        "knowledge, key constraints and boundary conditions, critical "
                        "definitions or formulas, and implicit assumptions. Focus on "
                        "assembling the factual and conceptual foundation, not solving."
                    ),
                },
                {
                    "name": "Agent C",
                    "system_prompt": (
                        f"You are Agent C (Executor). {_AGENT_C_CTX} "
                        "Solve the task step by step with clear reasoning."
                    ),
                },
            ]

        attn_impl = self.model_config.get("attn_implementation")
        trust_remote_code = bool(self.model_config.get("trust_remote_code", True))
        first = LLMAgent(
            model_name=self.model_config["name"],
            agent_id=0,
            role=roles[0]["name"],
            system_prompt=roles[0]["system_prompt"],
            dtype=self.model_config.get("dtype", "bfloat16"),
            trust_remote_code=trust_remote_code,
            device=device,
            attn_implementation=attn_impl,
            device_map=self.model_config.get("device_map"),
            max_memory=self.model_config.get("max_memory"),
        )
        self.agents.append(first)
        shared_model = first.model
        shared_tokenizer = first.tokenizer
        for i in range(1, len(roles)):
            agent = LLMAgent(
                model_name=self.model_config["name"],
                agent_id=i,
                role=roles[i]["name"],
                system_prompt=roles[i]["system_prompt"],
                dtype=self.model_config.get("dtype", "bfloat16"),
                trust_remote_code=trust_remote_code,
                device=device,
                attn_implementation=attn_impl,
                shared_model=shared_model,
                shared_tokenizer=shared_tokenizer,
            )
            self.agents.append(agent)

        ds_key = ""
        if dataset:
            from src.data.datasets import resolve_dataset_name

            ds_key = resolve_dataset_name(dataset.split(",")[0].strip())
        self.generate_bs = max(
            1,
            self.config.get("evaluation", {}).get("generate_bs", 1),
        )
        logger.info(
            "[TextMAS] dataset=%s roles_key=%s; %d agents (eval generate_bs=%d).",
            dataset or "(none)",
            ds_key or "(default roles)",
            len(self.agents),
            self.generate_bs,
        )

    def solve(self, question: str) -> dict[str, Any]:
        return self.solve_batch([question])[0]

    def solve_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        """Batched generation: Agent A & B in parallel, then Agent C per round.

        Sender agents (A / B) are pinned to ``_SENDER_MAX_NEW_TOKENS`` regardless
        of the dataset's own ``model.max_new_tokens``; only the executor (Agent C)
        gets the full per-dataset budget.
        """
        if not questions:
            return []
        num_rounds = self.method_config.get("num_rounds", 1)
        sampling_kwargs = dict(
            temperature=self.model_config.get("temperature", 0.7),
            top_p=self.model_config.get("top_p", 0.95),
            do_sample=self.model_config.get("do_sample", True),
        )
        sender_gen_kwargs = dict(
            max_new_tokens=_SENDER_MAX_NEW_TOKENS,
            **sampling_kwargs,
        )
        executor_gen_kwargs = dict(
            max_new_tokens=self.model_config.get("max_new_tokens", 1024),
            **sampling_kwargs,
        )

        n = len(questions)
        if len(self.agents) < 3:
            raise ValueError(
                "TextMAS pipeline requires at least 3 agents "
                "(Agent A, Agent B, Agent C)."
            )

        agent_a, agent_b, agent_c = self.agents[0], self.agents[1], self.agents[2]
        batch_t0 = time.time()

        def _merge_tu(acc: dict[str, int], tu: dict[str, int]) -> dict[str, int]:
            return {
                "prompt_tokens": acc["prompt_tokens"] + tu["prompt_tokens"],
                "completion_tokens": acc["completion_tokens"]
                + tu["completion_tokens"],
                "total_tokens": acc["total_tokens"] + tu["total_tokens"],
            }

        def _zero_usage() -> dict[str, int]:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # Agent A (Strategist) — all questions, empty context
        a_triples = agent_a.generate_batch_with_truncation_info(
            questions, context="", **sender_gen_kwargs
        )
        # Agent B (Knowledge Extractor) — all questions, empty context
        b_triples = agent_b.generate_batch_with_truncation_info(
            questions, context="", **sender_gen_kwargs
        )

        logs: list[list[dict[str, str]]] = []
        usage_a: list[dict[str, int]] = []
        usage_b: list[dict[str, int]] = []
        usage_c: list[dict[str, int]] = [_zero_usage() for _ in range(n)]
        truncated_flags: list[bool] = []

        for i in range(n):
            text_a, tr_a, tu_a = a_triples[i]
            text_b, tr_b, tu_b = b_triples[i]
            log_i: list[dict[str, str]] = [
                {"agent": agent_a.role, "round": 0, "text": text_a},
                {"agent": agent_b.role, "round": 0, "text": text_b},
            ]
            logs.append(log_i)
            usage_a.append(tu_a)
            usage_b.append(tu_b)
            truncated_flags.append(tr_a or tr_b)

        # Agent C (Executor) — receives Agent A + B text as context
        for round_idx in range(1, num_rounds + 1):
            contexts = [self._build_context(logs[i]) for i in range(n)]
            c_triples = agent_c.generate_batch_with_truncation_info(
                questions, contexts=contexts, **executor_gen_kwargs
            )
            for i in range(n):
                text_c, tr_c, tu_c = c_triples[i]
                logs[i].append(
                    {"agent": agent_c.role, "round": round_idx, "text": text_c}
                )
                truncated_flags[i] = truncated_flags[i] or tr_c
                usage_c[i] = _merge_tu(usage_c[i], tu_c)

        batch_elapsed = time.time() - batch_t0
        batch_mean = batch_elapsed / n if n else 0.0

        out: list[dict[str, Any]] = []
        for i in range(n):
            token_usage: dict[str, Any] = _merge_tu(
                _merge_tu(usage_a[i], usage_b[i]), usage_c[i]
            )
            token_usage["per_agent"] = {
                agent_a.role: usage_a[i],
                agent_b.role: usage_b[i],
                agent_c.role: usage_c[i],
            }
            out.append(
                {
                    "answer": logs[i][-1]["text"],
                    "reasoning": self._build_context(logs[i]),
                    "metadata": {
                        "method": "textmas",
                        "num_agents": len(self.agents),
                        "num_rounds": num_rounds,
                        "conversation_log": logs[i],
                        "wall_time": batch_mean,
                        "batch_mean_wall_time": batch_mean,
                        "batch_size": n,
                        "batched": True,
                        "truncated_by_max_new_tokens": truncated_flags[i],
                        "token_usage": token_usage,
                    },
                }
            )
        return out

    @staticmethod
    def _build_context(log: list[dict[str, str]]) -> str:
        parts = []
        for entry in log:
            parts.append(
                f"[{entry['agent']} | Round {entry['round']}]\n{entry['text']}"
            )
        return "\n\n---\n\n".join(parts)
