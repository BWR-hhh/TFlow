"""Inference configurations for ``baseline`` / ``textmas`` / ``tflow``.

This module replaces the previous ``configs/*.yaml`` hierarchy.  Each
public dict in :data:`CONFIGS` is the resolved (deep-merged) configuration
for one method.  Keys mirror the original YAML schema exactly so callers
in ``src/methods/*.py`` and ``src/evaluation/`` are unchanged.

Use :func:`get_config` to obtain a freshly deep-copied config; mutating
the returned dict will not affect any other caller.
"""

from __future__ import annotations

import copy
from typing import Dict


# ---------------------------------------------------------------------------
# Shared building blocks
# ---------------------------------------------------------------------------

# Sender-agent prompts shared by both TFlow's default and per-dataset entries.
# They must match the prompts used during training: deviating prompts shift
# the condition distribution and degrade the receiver agent's outputs.
_AGENT_A_PROMPT = (
    "You are Agent A (Strategist). Analyze HOW to approach this task: "
    "identify the type of reasoning required (deductive, inductive, "
    "abductive, analogical), plan the optimal chain-of-thought structure "
    "and depth, and anticipate common pitfalls and failure modes. Focus "
    "on the meta-cognitive strategy, not the answer itself."
)
_AGENT_B_PROMPT = (
    "You are Agent B (Knowledge Extractor). Identify WHAT knowledge is "
    "needed: the relevant domain(s) and specialized knowledge, key "
    "constraints and boundary conditions, critical definitions or "
    "formulas, and implicit assumptions. Focus on assembling the factual "
    "and conceptual foundation, not solving."
)


# ---------------------------------------------------------------------------
# Base configuration shared by all methods
# ---------------------------------------------------------------------------

_BASE: dict = {
    "seed": 42,
    "model": {
        "name": "Qwen/Qwen3-4B",
        "dtype": "bfloat16",
        "trust_remote_code": True,
        "max_new_tokens": 8192,
        "max_new_tokens_by_dataset": {
            "gsm8k": 2048,
            "minerva_math": 2048,
            "mmlu": 1024,
            "humanevalplus": 2048,
            "mbppplus": 2048,
        },
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample": True,
        "attn_implementation": "sdpa",
        # ``device_map: "auto"`` shards the model across all visible CUDA
        # devices via accelerate; pin to ``"cuda:0"`` for a single GPU.
        "device_map": "auto",
        "max_memory": None,
    },
    "data": {
        # Five supported benchmarks (canonical key + accepted aliases):
        # gsm8k, mbppplus (mbpp), humanevalplus (humaneval), minerva_math, mmlu.
        "dataset": "gsm8k",
        "split": "test",
        "max_samples": -1,
        # null -> Hugging Face default (HF_HOME / ~/.cache/huggingface).
        "cache_dir": None,
    },
    "evaluation": {
        # ``generate_bs > 1`` enables true batched inference for all methods.
        "generate_bs": 20,
        "extract_answer_regex": r"####\s*([\-\d\.\,]+)",
        "extract_answer_regex_by_dataset": {
            "minerva_math": r"\\boxed\{([^}]+)\}",
        },
        "metrics": ["accuracy", "token_cost", "latency"],
    },
    # Single-agent baseline picks ``system_prompt_by_dataset[<canonical key>]``
    # when available, otherwise falls back to ``system_prompt``.
    "method_config": {
        "system_prompt": (
            "You are Agent C (Executor). Solve the task step by step with "
            "clear reasoning, applying the optimal strategy and leveraging "
            "essential domain knowledge. Write each intermediate result "
            "explicitly. End with \\boxed{answer}."
        ),
        "system_prompt_by_dataset": {
            "gsm8k": (
                "Solve one step at a time, writing each intermediate result "
                "explicitly. Write numbers without commas (e.g. 57500 not "
                "57,500). End with \\boxed{answer}."
            ),
            "minerva_math": (
                "Solve this problem step by step with clear reasoning. Show "
                "all intermediate calculations. If the problem involves "
                "physical quantities, state units and verify dimensional "
                "consistency. Re-read the question before giving your final "
                "answer. Put your final answer inside \\boxed{}."
            ),
            "mmlu": (
                "Read the question and all options carefully. Apply the "
                "relevant domain knowledge to reason through each option. "
                "State why the correct answer is right and briefly note why "
                "the strongest distractor is wrong. End with \\boxed{X} "
                "where X is one of A, B, C, D."
            ),
            "humanevalplus": (
                "Output exactly one ```python``` block with the complete "
                "function. No extra text outside the code block."
            ),
            "mbppplus": (
                "Output exactly one ```python``` block. No test code or "
                "print statements."
            ),
        },
    },
    "logging": {
        "use_wandb": False,
        "project": "tflow",
        "entity": None,
        "run_name": None,
        "log_every": 10,
        # null -> ./outputs under the current working directory.
        "output_dir": None,
    },
}


# ---------------------------------------------------------------------------
# Method-specific overrides (deep-merged onto :data:`_BASE`)
# ---------------------------------------------------------------------------

_BASELINE_OVERRIDE: dict = {
    "method": "baseline",
}


_TEXTMAS_OVERRIDE: dict = {
    "method": "textmas",
    "method_config": {
        "num_agents": 3,
        "num_rounds": 1,
        "roles": [
            {
                "name": "Solver",
                "system_prompt": (
                    "You are a math agent. Given the final answer inside "
                    "\\boxed{YOUR_FINAL_ANSWER}."
                ),
            },
            {
                "name": "Critic",
                "system_prompt": (
                    "You are a science agent. Given the final answer "
                    "inside \\boxed{YOUR_FINAL_ANSWER}."
                ),
            },
            {
                "name": "Summarizer",
                "system_prompt": (
                    "You are a task summarizer. Given the final answer "
                    "inside \\boxed{YOUR_FINAL_ANSWER}."
                ),
            },
        ],
        "roles_by_dataset": {
            "gsm8k": [
                {
                    "name": "Solver",
                    "system_prompt": (
                        "You are the Solver (Executor). Solve one step at a "
                        "time, writing each intermediate result explicitly. "
                        "Write numbers without commas (e.g. 57500 not 57,500). "
                        "End with \\boxed{answer}."
                    ),
                },
                {
                    "name": "Critic",
                    "system_prompt": (
                        "You are the Critic (Strategist + Knowledge role). "
                        "You do not see any other agent's output-only the "
                        "problem. Independently list the relevant domain, "
                        "definitions/formulas, constraints (units, "
                        "integrality, ranges), and common pitfalls a correct "
                        "solution must respect. You may sketch your own brief "
                        "approach. Do not refer to or assume a Solver "
                        "transcript."
                    ),
                },
                {
                    "name": "Summarizer",
                    "system_prompt": (
                        "You are the Summarizer (final Executor). Reconcile "
                        "the Solver and Critic transcripts. Solve one step "
                        "at a time, writing each intermediate result "
                        "explicitly. Write numbers without commas (e.g. 57500 "
                        "not 57,500). End with \\boxed{answer}."
                    ),
                },
            ],
            "minerva_math": [
                {
                    "name": "Solver",
                    "system_prompt": (
                        "You are a math agent. Given the final answer "
                        "inside \\boxed{YOUR_FINAL_ANSWER}."
                    ),
                },
                {
                    "name": "Critic",
                    "system_prompt": (
                        "You are a science agent. Given the final answer "
                        "inside \\boxed{YOUR_FINAL_ANSWER}."
                    ),
                },
                {
                    "name": "Summarizer",
                    "system_prompt": (
                        "You are a task summarizer. Given the final answer "
                        "inside \\boxed{YOUR_FINAL_ANSWER}."
                    ),
                },
            ],
            "mmlu": [
                {
                    "name": "Solver",
                    "system_prompt": (
                        "You are the Solver (Executor). Given the final "
                        "answer inside \\boxed{YOUR_FINAL_ANSWER}."
                    ),
                },
                {
                    "name": "Critic",
                    "system_prompt": (
                        "You are the Critic (Strategist + Knowledge role). "
                        "Given the final answer inside \\boxed{YOUR_FINAL_ANSWER}."
                    ),
                },
                {
                    "name": "Summarizer",
                    "system_prompt": (
                        "You are the Summarizer (final Executor). Given the "
                        "final answer inside \\boxed{YOUR_FINAL_ANSWER}."
                    ),
                },
            ],
            "humanevalplus": [
                {
                    "name": "Solver",
                    "system_prompt": (
                        "You are a math agent. Output python code as "
                        "self-contained Python function."
                    ),
                },
                {
                    "name": "Critic",
                    "system_prompt": (
                        "You are a science agent. Output python code as "
                        "self-contained Python function."
                    ),
                },
                {
                    "name": "Summarizer",
                    "system_prompt": (
                        "You are a task summarizer. Given the final answer "
                        "in markdown python code block."
                    ),
                },
            ],
            "mbppplus": [
                {
                    "name": "Solver",
                    "system_prompt": (
                        "You are a math agent. You must put all python code "
                        "as self-contained Python function in markdown code "
                        "blocks."
                    ),
                },
                {
                    "name": "Critic",
                    "system_prompt": (
                        "You are a science agent. You must put all python "
                        "code as self-contained Python function in markdown "
                        "code blocks."
                    ),
                },
                {
                    "name": "Summarizer",
                    "system_prompt": (
                        "You are a task summarizer. Given the final answer "
                        "in markdown python code block."
                    ),
                },
            ],
        },
        "aggregation": "last_agent",
    },
}


_TFLOW_OVERRIDE: dict = {
    # TFlow runs hidden-state extraction and LoRA-hook injection on the same
    # backbone instance; pinning ``device_map`` keeps the activations local.
    "model": {
        "device_map": "cuda:0",
        "attn_implementation": "sdpa",
    },
    "method": "tflow",
    "method_config": {
        "num_agents": 3,
        # Agents A and B are the ParameterGenerator's senders (their hidden
        # states form the condition input).  At evaluation time the system
        # prompts must match those used during training.
        "agent_prompts": [
            _AGENT_A_PROMPT,
            _AGENT_B_PROMPT,
            (
                "You are Agent C (Executor). Solve the task step by step "
                "with clear reasoning."
            ),
        ],
        "agent_prompts_by_dataset": {
            "gsm8k": [
                _AGENT_A_PROMPT,
                _AGENT_B_PROMPT,
                (
                    "You are Agent C (Executor). Solve with concise steps: "
                    "write each intermediate numeric result once; do not "
                    "repeat the same scenario or recalculate the same chain "
                    "in multiple repeated blocks. Before computing, re read "
                    "the question to confirm exactly which quantity is asked "
                    "and the correct time unit (day/week/month/year). For "
                    "recycling or merge problems (e.g. empties combined "
                    "into new units), iterate until the stop rule and answer "
                    "what the question asks (e.g. total pens used). After "
                    "reaching your answer, verify that every piece of given "
                    "information was used; if something was ignored, "
                    "reconsider your interpretation. Output the final "
                    "numerical answer in a single line as \\boxed{answer} "
                    "with digits only and no commas (e.g. 57500 not 57,500). "
                    "End your message with that one \\boxed{} line."
                ),
            ],
            "minerva_math": [
                _AGENT_A_PROMPT,
                _AGENT_B_PROMPT,
                (
                    "You are Agent C (Executor). Solve concisely: state the "
                    "key formula, convert ALL units to a consistent system "
                    "before computing (Mpc to pc, km to cm, etc.), and state "
                    "each physical constant with its value and units. "
                    "Compute once; never repeat a calculation or revisit an "
                    "approach already tried. Output your final answer "
                    "inside \\boxed{}. End with that one \\boxed{} line."
                ),
            ],
            "mmlu": [
                _AGENT_A_PROMPT,
                _AGENT_B_PROMPT,
                (
                    "You are Agent C (Executor). Read the question and all "
                    "options carefully. Apply the relevant domain knowledge "
                    "to reason through each option. State why the correct "
                    "answer is right and briefly note why the strongest "
                    "distractor is wrong. End with \\boxed{X} where X is "
                    "one of A, B, C, D."
                ),
            ],
            "humanevalplus": [
                _AGENT_A_PROMPT,
                _AGENT_B_PROMPT,
                (
                    "You are Agent C (Executor). Output exactly one "
                    "```python``` block containing the complete function. "
                    "Keep reasoning minimal: identify the core algorithm, "
                    "then write code immediately. NEVER trace through test "
                    "cases, enumerate examples, or simulate execution. If a "
                    "helper function is provided, use it directly. Handle "
                    "edge cases (n=0, empty input). Use `/` not `//` unless "
                    "integer division is explicitly required. No text after "
                    "the code block."
                ),
            ],
            "mbppplus": [
                _AGENT_A_PROMPT,
                _AGENT_B_PROMPT,
                (
                    "You are Agent C (Executor). Output exactly one "
                    "```python``` block containing the complete function. "
                    "Keep reasoning minimal: identify the core algorithm, "
                    "then write code immediately. NEVER trace through test "
                    "cases, enumerate examples, or simulate execution. "
                    "Match the exact function name and signature from the "
                    "asserts. Use `/` not `//` unless integer division is "
                    "explicitly required. No text after the code block."
                ),
            ],
        },
        # ``--checkpoint`` takes precedence over ``eval_checkpoint``.
        "training": {
            "checkpoint_dir": "checkpoints/tflow",
            "eval_checkpoint": None,
        },
    },
}


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` onto a deep copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


CONFIGS: Dict[str, dict] = {
    "baseline": _deep_merge(_BASE, _BASELINE_OVERRIDE),
    "textmas": _deep_merge(_BASE, _TEXTMAS_OVERRIDE),
    "tflow": _deep_merge(_BASE, _TFLOW_OVERRIDE),
}

AVAILABLE_METHODS: tuple = tuple(CONFIGS.keys())


def get_config(method: str) -> dict:
    """Return a freshly deep-copied configuration for ``method``."""
    if method not in CONFIGS:
        raise ValueError(
            f"Unknown method: {method!r}. Choose from {list(CONFIGS.keys())}."
        )
    return copy.deepcopy(CONFIGS[method])
