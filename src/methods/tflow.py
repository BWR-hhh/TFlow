"""TFlow (Thought Flow): weight-space inter-agent communication.

This module implements the inference path of TFlow.  Sender agents A and B
encode the question into multi-layer hidden states; a learned per-layer
softmax aggregates these into a condition tensor that drives the
``ParameterGenerator``, which emits a LoRA factor pair ``(A, B)`` for
every targeted ``nn.Linear`` of the backbone.  The per-sender factor
pairs are convex-combined and applied to the receiver agent C through
per-sample forward hooks for the final generation pass.

The ``ParameterGenerator`` architecture and the runtime fusion knobs (PG
depth, condition extraction strategy, fusion weights, dimensions, …) are
fixed as module-level constants below; they are part of the model
definition.  The YAML configuration only exposes the upstream backbone,
agent system prompts, and the evaluation checkpoint pointer.
"""

from __future__ import annotations

import os
import re
import time
import logging
from typing import Any

import torch
import torch.nn as nn

from src.methods.base import BaseMethod
from src.models.agent import LLMAgent
from src.models.param_gen import ParameterGenerator, ParameterGeneratorConfig
from src.models.param_gen.backbone_pg_mapping import infer_pg_mapping_from_causal_lm
from src.models.param_gen.tflow_bridge import (
    build_weighted_lora_entries,
    pg_state_dict_to_batched_lora_factors,
    pg_state_dict_to_lora_factors,
)
from src.models.lora import (
    apply_batched_multi_lora_hooks,
    remove_batched_multi_lora_hooks,
    patch_linear_multi_lora_factors,
    restore_forward_patches,
)
from src.utils.helpers import checkpoint_backbone_tag, get_device

logger = logging.getLogger(__name__)


# ============================================================================
# ParameterGenerator architecture and runtime fusion knobs.
#
# These constants pin the PG's depth, condition extraction strategy, fusion
# rule, and dimensions; they are part of the model definition and are not
# exposed to YAML.  Only the upstream backbone (``model.*``) and the
# evaluation checkpoint (``method_config.training.eval_checkpoint``) are
# user-configurable.
# ============================================================================

# The ParameterGenerator is constructed locally from these constants (no
# pretrained weights are downloaded), so the evaluation checkpoint must
# always supply the trained PG parameters.
_PG_TORCH_DTYPE = torch.bfloat16

# Convex-combination weights for the senders' LoRA factors.  When the
# length does not match the number of senders, the weights are silently
# re-balanced to uniform 1/n.
_FIXED_SENDER_WEIGHTS = (0.5, 0.5)

# Condition extraction: keep the raw token sequence, aggregate every decoder
# hidden state via a learnable softmax (``LayerAggregator``), and pass only
# the question tokens to the PG.
_PG_CONDITION_QUESTION_ONLY = True
_PG_LAYER_POOL_TEMPERATURE = 1.0

# PG forward flags (set after construction).
_PG_USE_COND_PROJ = False
_PG_USE_COND_TOKENS = True
_PG_COND_TOKENS_MODE = "replace"

# pg_mapping is inferred per-backbone by enumerating every 2-D nn.Linear
# under the first decoder layer.
_PG_MAPPING_ALL_LINEARS = True

# Compact PG architecture; sized for Qwen3-4B/8B-class backbones (36 layers).
_PG_ARCH = {
    "d_model": 1024,
    "head_dim": 128,
    "num_pg_layers": 2,
    "output_dim": 256,
    "token_dim": 256,
    "rank": 4,
    "alpha": 8,
    "dim_accumulation": 2,
    "num_base_model_layers": 36,
}


_DEFAULT_TFLOW_AGENT_PROMPTS = [
    (
        "You are Agent A (Strategist). Analyze HOW to approach this task: identify the type of "
        "reasoning required (deductive, inductive, abductive, analogical), plan the optimal "
        "chain-of-thought structure and depth, and anticipate common pitfalls and failure modes. "
        "Focus on the meta-cognitive strategy, not the answer itself."
    ),
    (
        "You are Agent B (Knowledge Extractor). Identify WHAT knowledge is needed: the relevant "
        "domain(s) and specialized knowledge, key constraints and boundary conditions, critical "
        "definitions or formulas, and implicit assumptions. Focus on assembling the factual and "
        "conceptual foundation, not solving."
    ),
    (
        "You are Agent C (Executor). You are the main solver. Solve the task step by step with "
        "clear reasoning, applying the optimal strategy and leveraging essential domain knowledge. "
        "Write each intermediate result explicitly. End with the final answer."
    ),
]


class LayerAggregator(nn.Module):
    """Learnable per-layer softmax over backbone hidden states (``weighted_pool``).

    Input: list of ``L = num_hidden_layers + 1`` tensors of shape ``(seq, D)``.
    Output: ``(seq, D)``.
    """

    def __init__(self, num_layers: int, hidden_size: int, *, temperature: float = 1.0):
        super().__init__()
        self.num_layers = num_layers
        self.output_dim = hidden_size
        self.layer_weights = nn.Parameter(torch.zeros(num_layers))
        self.log_temperature = nn.Parameter(torch.tensor(float(temperature)).log())

    def forward(self, layer_hiddens: list[torch.Tensor]) -> torch.Tensor:
        temp = self.log_temperature.exp().clamp(min=0.01)
        w = torch.softmax(self.layer_weights / temp, dim=0)
        stacked = torch.stack(layer_hiddens, dim=0)
        return (w[:, None, None] * stacked).sum(dim=0)


class TFlowMethod(BaseMethod):
    """TFlow (Thought Flow) — ParameterGenerator adapter.

    Pipeline (per question, ``num_agents`` agents share one backbone):
      1. Each sender agent encodes the question, exposing the per-layer
         hidden states at the question tokens.
      2. ``LayerAggregator`` reduces the per-layer hidden states to one
         tensor through a learnable softmax over layers.
      3. ``ParameterGenerator`` maps each sender's aggregated tensor to a
         pair of LoRA factors ``(A, B)`` per targeted ``nn.Linear``.
      4. The senders' factor pairs are convex-combined with
         :data:`_FIXED_SENDER_WEIGHTS`.
      5. The receiver agent generates the final answer with the combined
         LoRA factors applied through per-sample forward hooks.
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self.agents: list[LLMAgent] = []
        self.device: torch.device = get_device()
        self._model_shared: bool = False
        # Populated by _load_checkpoint from the torch.save key "samples_seen".
        self.checkpoint_samples_seen: int = 0
        self.parameter_generator: ParameterGenerator | None = None
        # Device chosen by _resolve_parameter_generator_device: prefer the
        # highest-index CUDA device that is not hosting a backbone shard.
        self.parameter_generator_device: torch.device | None = None
        self._fixed_sender_weights: list[float] = []
        self.pg_layer_aggregator: LayerAggregator | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _resolve_parameter_generator_device(self) -> torch.device:
        """Pick the highest-index CUDA device that does not host a backbone parameter.

        Falls back to ``self.device`` when only one GPU is visible.
        """
        if not torch.cuda.is_available():
            return self.device
        used: set[int] = set()
        try:
            for p in self.agents[0].model.parameters():
                if p.device.type == "cuda" and p.device.index is not None:
                    used.add(int(p.device.index))
        except Exception:
            pass
        for idx in range(torch.cuda.device_count() - 1, -1, -1):
            if idx not in used:
                dev = torch.device(f"cuda:{idx}")
                if dev != self.device:
                    logger.info(
                        "[TFlow] ParameterGenerator → %s (auto: backbone uses %s).",
                        dev,
                        sorted(used) if used else "(n/a)",
                    )
                return dev
        return self.device

    def _resolve_agent_prompts_raw(self) -> list | None:
        """Prefer ``agent_prompts_by_dataset[data.dataset]`` when set, else ``agent_prompts``."""
        by_ds = self.method_config.get("agent_prompts_by_dataset") or {}
        dataset = self.config.get("data", {}).get("dataset", "")
        if dataset and by_ds:
            from src.data.datasets import resolve_dataset_name

            key = resolve_dataset_name(dataset.split(",")[0].strip())
            if key in by_ds:
                raw = by_ds[key]
                return list(raw) if raw is not None else None
        return self.method_config.get("agent_prompts")

    def _apply_tflow_eval_mode(self) -> None:
        """Pin ParameterGenerator + LayerAggregator + shared LLM to eval mode."""
        assert self.parameter_generator is not None
        self.parameter_generator.eval()
        if self.pg_layer_aggregator is not None:
            self.pg_layer_aggregator.eval()
        for agent in self.agents:
            agent.model.eval()

    def _ensure_tflow_inference_eval_mode(self) -> None:
        if self.parameter_generator is None:
            return
        self._apply_tflow_eval_mode()

    def setup(self) -> None:
        """Inference setup: build agents + ParameterGenerator and load eval checkpoint."""
        num_agents = self.method_config.get("num_agents", 3)

        raw_prompts = self._resolve_agent_prompts_raw()
        if raw_prompts is None:
            agent_prompts = list(_DEFAULT_TFLOW_AGENT_PROMPTS)
        else:
            agent_prompts = [str(p).strip() for p in raw_prompts if str(p).strip()]
        if len(agent_prompts) < num_agents:
            logger.warning(
                f"agent_prompts has {len(agent_prompts)} entries but num_agents={num_agents}; "
                "cycling prompts."
            )
            while len(agent_prompts) < num_agents:
                agent_prompts.extend(_DEFAULT_TFLOW_AGENT_PROMPTS)
            agent_prompts = agent_prompts[:num_agents]
        elif len(agent_prompts) > num_agents:
            agent_prompts = agent_prompts[:num_agents]

        # --- Agents: one shared backbone, distinct system prompts ---
        attn_impl = self.model_config.get("attn_implementation")
        first_agent = LLMAgent(
            model_name=self.model_config["name"],
            agent_id=0,
            role="agent_0",
            system_prompt=agent_prompts[0],
            dtype=self.model_config.get("dtype", "bfloat16"),
            device=self.device,
            attn_implementation=attn_impl,
            device_map=self.model_config.get("device_map"),
            max_memory=self.model_config.get("max_memory"),
        )
        self.agents.append(first_agent)

        shared_model = first_agent.model
        shared_tokenizer = first_agent.tokenizer
        for i in range(1, num_agents):
            agent = LLMAgent(
                model_name=self.model_config["name"],
                agent_id=i,
                role=f"agent_{i}",
                system_prompt=agent_prompts[i],
                dtype=self.model_config.get("dtype", "bfloat16"),
                device=self.device,
                attn_implementation=attn_impl,
                shared_model=shared_model,
                shared_tokenizer=shared_tokenizer,
            )
            self.agents.append(agent)
        self._model_shared = True

        ds_key = ""
        dataset = self.config.get("data", {}).get("dataset", "")
        if dataset:
            from src.data.datasets import resolve_dataset_name

            ds_key = resolve_dataset_name(dataset.split(",")[0].strip())
        logger.info(
            "[TFlow] dataset=%s agent_prompts_key=%s; prompts: %s",
            dataset or "(none)",
            ds_key or "(default)",
            "; ".join(f"{j}={agent_prompts[j][:60]}..." for j in range(num_agents)),
        )

        hidden_size = self.agents[0].hidden_size
        num_model_layers = self.agents[0].model.config.num_hidden_layers
        ckpt_dir_cfg = (
            self.method_config.get("training", {}).get("checkpoint_dir")
            or "checkpoints/tflow"
        )
        ckpt_dir = os.path.abspath(os.path.expanduser(str(ckpt_dir_cfg)))

        self._setup_parameter_generator(
            num_agents=num_agents,
            hidden_size=hidden_size,
            num_model_layers=num_model_layers,
            ckpt_dir=ckpt_dir,
        )
        self._apply_tflow_eval_mode()

        logger.info(
            "[TFlow] Setup complete (eval): %d agents, PG input_dim=%s",
            num_agents,
            self.parameter_generator.config.input_dim,
        )

    def _setup_parameter_generator(
        self,
        *,
        num_agents: int,
        hidden_size: int,
        num_model_layers: int,
        ckpt_dir: str,
    ) -> None:
        # --- Sender fusion weights (uniform fallback when length mismatches) ---
        num_senders = max(num_agents - 1, 0)
        if num_senders == 0:
            self._fixed_sender_weights = []
        elif len(_FIXED_SENDER_WEIGHTS) == num_senders:
            self._fixed_sender_weights = list(_FIXED_SENDER_WEIGHTS)
        else:
            logger.warning(
                "[TFlow] _FIXED_SENDER_WEIGHTS has len %d but num_senders=%d; using equal 1/n.",
                len(_FIXED_SENDER_WEIGHTS),
                num_senders,
            )
            self._fixed_sender_weights = [1.0 / num_senders] * num_senders
        s = sum(self._fixed_sender_weights)
        if num_senders > 0 and abs(s - 1.0) > 1e-3:
            self._fixed_sender_weights = [w / s for w in self._fixed_sender_weights]

        self.parameter_generator_device = self._resolve_parameter_generator_device()
        pg_dev = self.parameter_generator_device

        # --- ParameterGenerator config: backbone-aware pg_mapping + fixed arch ---
        pg_mapping = infer_pg_mapping_from_causal_lm(
            self.agents[0].model,
            include_substrings=None,
            all_linears=_PG_MAPPING_ALL_LINEARS,
        )
        pg_cfg = ParameterGeneratorConfig(
            input_dim=int(hidden_size),
            pg_mapping=pg_mapping,
            **_PG_ARCH,
        )

        if int(pg_cfg.num_base_model_layers) != int(num_model_layers):
            logger.warning(
                "[TFlow] _PG_ARCH.num_base_model_layers=%s != backbone num_hidden_layers=%s — "
                "PG LoRA layout assumes the backbone has %s layers.",
                pg_cfg.num_base_model_layers,
                num_model_layers,
                pg_cfg.num_base_model_layers,
            )

        # Local construction (no HF download / no pretrained weight load).
        self.parameter_generator = ParameterGenerator(pg_cfg)
        self.parameter_generator.to(device=pg_dev, dtype=_PG_TORCH_DTYPE)

        pg_model = self.parameter_generator.model
        pg_model.use_cond_proj = _PG_USE_COND_PROJ
        pg_model.use_cond_tokens = _PG_USE_COND_TOKENS
        pg_model.cond_tokens_mode = _PG_COND_TOKENS_MODE

        # --- Layer aggregator (weighted_pool over decoder hidden states) ---
        total_layers = int(num_model_layers) + 1
        self.pg_layer_aggregator = LayerAggregator(
            num_layers=total_layers,
            hidden_size=int(hidden_size),
            temperature=_PG_LAYER_POOL_TEMPERATURE,
        ).to(pg_dev)

        for p in self.parameter_generator.parameters():
            p.requires_grad_(False)
        self.parameter_generator.eval()

        self._load_checkpoint(ckpt_dir)

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _resolve_ckpt_path(self, ckpt_dir: str, user_path: str) -> str:
        if os.path.isfile(user_path):
            return user_path
        base = os.path.basename(user_path)
        ckpt_path = os.path.join(ckpt_dir, base)
        if not os.path.exists(ckpt_path):
            ds = self.config.get("data", {}).get("dataset", "unknown")
            ds_tag = re.sub(r"[^\w\-]", "_", str(ds)) or "unknown"
            bb_tag = checkpoint_backbone_tag(self.config)
            alt_bb = os.path.join(ckpt_dir, f"{ds_tag}_{bb_tag}_{base}")
            alt = os.path.join(ckpt_dir, f"{ds_tag}_{base}")
            if os.path.exists(alt_bb):
                ckpt_path = alt_bb
            elif os.path.exists(alt):
                ckpt_path = alt
        return ckpt_path

    def _load_checkpoint(self, ckpt_dir: str) -> None:
        """Resolve and load the evaluation checkpoint.

        Resolution order:

        1. ``method_config.training.eval_checkpoint`` (overridable by
           ``--checkpoint``);
        2. ``{dataset}_{backbone}_tflow_best.pt`` under ``ckpt_dir``;
        3. ``{dataset}_tflow_best.pt`` under ``ckpt_dir``;
        4. ``tflow_best.pt`` under ``ckpt_dir``.

        If none exists, the ParameterGenerator keeps its random
        initialisation; the run is then only a smoke test of the data flow.
        """
        if not ckpt_dir:
            return

        train_cfg = self.method_config.get("training", {})
        eval_ckpt = train_cfg.get("eval_checkpoint") or train_cfg.get("checkpoint_file")

        ckpt_path: str | None = None
        if eval_ckpt:
            ckpt_path = self._resolve_ckpt_path(ckpt_dir, eval_ckpt)
            if not os.path.exists(ckpt_path):
                logger.warning(
                    f"eval_checkpoint not found: {ckpt_path}, falling back to best."
                )
                ckpt_path = None
        if not ckpt_path:
            ds = self.config.get("data", {}).get("dataset", "unknown")
            ds_tag = re.sub(r"[^\w\-]", "_", str(ds)) or "unknown"
            bb_tag = checkpoint_backbone_tag(self.config)
            best_ds_bb = os.path.join(ckpt_dir, f"{ds_tag}_{bb_tag}_tflow_best.pt")
            best_ds = os.path.join(ckpt_dir, f"{ds_tag}_tflow_best.pt")
            best_legacy = os.path.join(ckpt_dir, "tflow_best.pt")
            if os.path.exists(best_ds_bb):
                ckpt_path = best_ds_bb
            elif os.path.exists(best_ds):
                ckpt_path = best_ds
            else:
                ckpt_path = best_legacy

        if not ckpt_path or not os.path.exists(ckpt_path):
            logger.warning(
                f"No eval checkpoint at {ckpt_path}. Using random init for ParameterGenerator."
            )
            return

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except TypeError:
            state = torch.load(ckpt_path, map_location="cpu")

        saved_flags = state.get("pg_config_flags")
        if saved_flags is not None:
            current_flags = {
                "use_cond_proj": _PG_USE_COND_PROJ,
                "use_cond_tokens": _PG_USE_COND_TOKENS,
                "cond_tokens_mode": _PG_COND_TOKENS_MODE,
            }
            mismatches = {
                k: (saved_flags[k], current_flags[k])
                for k in saved_flags
                if k in current_flags and saved_flags[k] != current_flags[k]
            }
            if mismatches:
                msg_lines = [
                    f"  {k}: checkpoint={sv}  fixed={cv}"
                    for k, (sv, cv) in mismatches.items()
                ]
                raise RuntimeError(
                    "Architecture constants disagree with the training checkpoint:\n"
                    + "\n".join(msg_lines)
                )
            logger.info("[TFlow] Evaluation flags match the checkpoint.")
        else:
            logger.warning(
                "[TFlow] Checkpoint has no pg_config_flags; configuration "
                "consistency cannot be verified."
            )
        if "parameter_generator" in state and self.parameter_generator is not None:
            _pg_strict = saved_flags is not None
            try:
                info = self.parameter_generator.load_state_dict(
                    state["parameter_generator"], strict=_pg_strict
                )
            except RuntimeError as e:
                raise RuntimeError(
                    "ParameterGenerator checkpoint does not match the current "
                    "_PG_ARCH (num_pg_layers / d_model / rank / ...).  The "
                    "training and evaluation runs must use the same constants."
                    f"\nOriginal error: {e}"
                ) from e
            if not _pg_strict and info.missing_keys:
                logger.warning(
                    "[TFlow] Legacy checkpoint; the following keys keep their "
                    "default initialisation: %s",
                    info.missing_keys,
                )
        if (
            "pg_layer_aggregator" in state
            and self.pg_layer_aggregator is not None
        ):
            try:
                self.pg_layer_aggregator.load_state_dict(
                    state["pg_layer_aggregator"], strict=True
                )
            except RuntimeError as e:
                raise RuntimeError(
                    "pg_layer_aggregator checkpoint does not match the current "
                    "configuration."
                    f"\nOriginal error: {e}"
                ) from e
            logger.info("[TFlow] Loaded pg_layer_aggregator state from checkpoint.")
        self.checkpoint_samples_seen = int(state.get("samples_seen", 0))
        logger.info(f"[TFlow] Loaded ParameterGenerator checkpoint from {ckpt_path}")

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _to_pg_dtype_dev(self, x: torch.Tensor) -> torch.Tensor:
        assert self.parameter_generator is not None
        pg_dtype = next(self.parameter_generator.parameters()).dtype
        pg_dev = self.parameter_generator_device or self.device
        return x.to(device=pg_dev, dtype=pg_dtype)

    def _per_sender_condition(self, agent: LLMAgent, question: str) -> torch.Tensor:
        """Question-only multi-layer hidden states → ``LayerAggregator`` → PG-ready cond ``(1, seq, hidden)``."""
        with torch.no_grad():
            all_layers = agent.extract_hidden_states_all_layers(
                question, question_only=_PG_CONDITION_QUESTION_ONLY
            )
        agg_hs = self.pg_layer_aggregator(all_layers)  # (seq, hidden)
        return self._to_pg_dtype_dev(agg_hs.unsqueeze(0))

    def _per_sender_condition_batch(
        self, agent: LLMAgent, questions: list[str]
    ) -> torch.Tensor:
        """Batched variant: returns ``(B, max_seq, hidden)`` zero-padded along seq."""
        batch_all_layers = agent.extract_hidden_states_all_layers_batch(
            questions, question_only=_PG_CONDITION_QUESTION_ONLY
        )
        per_sample = [
            self.pg_layer_aggregator(sample_layers).unsqueeze(0)
            for sample_layers in batch_all_layers
        ]
        max_len = max(c.shape[1] for c in per_sample)
        padded = []
        for c in per_sample:
            pad_len = max_len - c.shape[1]
            if pad_len > 0:
                c = torch.nn.functional.pad(c, (0, 0, 0, pad_len))
            padded.append(c)
        return self._to_pg_dtype_dev(torch.cat(padded, dim=0))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def solve(self, question: str) -> dict[str, Any]:
        """Senders → ParameterGenerator → fused LoRA factors → receiver generates."""
        self._ensure_tflow_inference_eval_mode()
        t0 = time.time()
        gen_kwargs = dict(
            max_new_tokens=self.model_config.get("max_new_tokens", 1024),
            temperature=self.model_config.get("temperature", 0.7),
            do_sample=self.model_config.get("do_sample", True),
        )

        n = len(self.agents)
        if n < 2:
            out, truncated, token_usage = self.agents[0].generate_with_truncation_info(
                question, **gen_kwargs
            )
            return {
                "answer": out,
                "reasoning": out,
                "metadata": {
                    "method": "tflow",
                    "num_agents": n,
                    "note": "single_agent_fallback",
                    "wall_time": time.time() - t0,
                    "truncated_by_max_new_tokens": truncated,
                    "token_usage": token_usage,
                },
            }

        senders = self.agents[:-1]
        receiver = self.agents[-1]

        factor_dicts: list[dict] = []
        for agent in senders:
            cond = self._per_sender_condition(agent, question)
            with torch.no_grad():
                pg_out = self.parameter_generator(cond)
            factor_dicts.append(pg_state_dict_to_lora_factors(pg_out))

        weights = list(self._fixed_sender_weights)
        weighted_entries = build_weighted_lora_entries(factor_dicts, weights)
        patches = patch_linear_multi_lora_factors(receiver.model, weighted_entries)
        try:
            final_answer, truncated, token_usage = receiver.generate_with_truncation_info(
                question, **gen_kwargs
            )
        finally:
            restore_forward_patches(patches)

        elapsed = time.time() - t0
        return {
            "answer": final_answer,
            "reasoning": final_answer,
            "metadata": {
                "method": "tflow",
                "num_agents": n,
                "receiver_index": n - 1,
                "num_senders": len(senders),
                "sender_weights": [float(w) for w in weights],
                "wall_time": elapsed,
                "truncated_by_max_new_tokens": truncated,
                "token_usage": token_usage,
            },
        }

    def solve_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        """Fully-batched TFlow inference via per-sample LoRA forward hooks."""
        if not questions:
            return []

        self._ensure_tflow_inference_eval_mode()
        t0_batch = time.time()

        gen_kwargs = dict(
            max_new_tokens=self.model_config.get("max_new_tokens", 1024),
            temperature=self.model_config.get("temperature", 0.7),
            do_sample=self.model_config.get("do_sample", True),
        )

        n = len(self.agents)
        if n < 2:
            triples = self.agents[0].generate_batch_with_truncation_info(
                questions, **gen_kwargs
            )
            per = (time.time() - t0_batch) / len(questions)
            return [
                {
                    "answer": out,
                    "reasoning": out,
                    "metadata": {
                        "method": "tflow",
                        "num_agents": n,
                        "note": "single_agent_fallback_batch",
                        "wall_time": per,
                        "truncated_by_max_new_tokens": truncated,
                        "token_usage": token_usage,
                    },
                }
                for out, truncated, token_usage in triples
            ]

        return self._solve_batch_pg(questions, gen_kwargs, t0_batch)

    @torch.no_grad()
    def _solve_batch_pg(
        self,
        questions: list[str],
        gen_kwargs: dict,
        t0_batch: float,
    ) -> list[dict[str, Any]]:
        assert self.parameter_generator is not None
        B = len(questions)
        senders = self.agents[:-1]
        receiver = self.agents[-1]
        n = len(self.agents)

        batched_factor_dicts: list[dict] = []
        for agent in senders:
            cond = self._per_sender_condition_batch(agent, questions)
            pg_out = self.parameter_generator(cond)
            batched_factor_dicts.append(pg_state_dict_to_batched_lora_factors(pg_out))

        weights = list(self._fixed_sender_weights)
        weighted_entries = build_weighted_lora_entries(batched_factor_dicts, weights)
        apply_batched_multi_lora_hooks(receiver.model, weighted_entries)
        try:
            answer_triples = receiver.generate_batch_with_truncation_info(
                questions, **gen_kwargs
            )
        finally:
            remove_batched_multi_lora_hooks(receiver.model)

        elapsed = time.time() - t0_batch
        per_time = elapsed / max(B, 1)
        weights_meta = [float(w) for w in weights]
        return [
            {
                "answer": ans,
                "reasoning": ans,
                "metadata": {
                    "method": "tflow",
                    "num_agents": n,
                    "receiver_index": n - 1,
                    "num_senders": len(senders),
                    "sender_weights": weights_meta,
                    "wall_time": per_time,
                    "batch_size": B,
                    "batched": True,
                    "truncated_by_max_new_tokens": truncated,
                    "token_usage": token_usage,
                },
            }
            for ans, truncated, token_usage in answer_triples
        ]
