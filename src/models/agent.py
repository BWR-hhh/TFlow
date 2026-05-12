"""Thin wrapper around a HuggingFace causal LM used as a single agent.

In addition to standard chat-style generation, the wrapper exposes the
multi-layer hidden states needed to drive the TFlow ParameterGenerator and
provides hooks for runtime LoRA injection.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional, Set

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class LLMAgent:
    """Single agent backed by a HuggingFace causal LM.

    Capabilities:

    - chat-style generation (``generate_*`` methods);
    - per-layer hidden-state extraction at the question span, used as the
      condition input for the TFlow ParameterGenerator;
    - LoRA forward-hook installation and removal via :mod:`src.models.lora`.
    """

    def __init__(
        self,
        model_name: str,
        agent_id: int = 0,
        role: str = "assistant",
        system_prompt: str = "You are a helpful assistant.",
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
        device: Optional[torch.device] = None,
        attn_implementation: Optional[str] = None,
        shared_model: Optional[nn.Module] = None,
        shared_tokenizer: Optional[AutoTokenizer] = None,
        device_map: Optional[Any] = None,
        max_memory: Optional[dict] = None,
    ):
        self.agent_id = agent_id
        self.role = role
        self.system_prompt = system_prompt
        self.model_name = model_name

        self.device = device or (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        if shared_model is not None and shared_tokenizer is not None:
            self.model = shared_model
            self.tokenizer = shared_tokenizer
            self.hidden_size = self.model.config.hidden_size
            self._model_input_device = getattr(
                self.model, "device", next(self.model.parameters()).device
            )
            logger.info(
                f"[Agent {agent_id}] Sharing model. hidden_size={self.hidden_size}, "
                f"anchor_device={self.device}, model_input_device={self._model_input_device}"
            )
            return

        torch_dtype = getattr(torch, dtype, torch.bfloat16)

        logger.info(f"[Agent {agent_id}] Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        load_kwargs: dict = dict(
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=True,
        )
        if attn_implementation:
            load_kwargs["attn_implementation"] = attn_implementation

        # device_map:
        # - Explicit "auto" / dict / "cuda:0": use as-is (HF + accelerate).
        # - None: if multiple CUDA devices are visible, default to "auto" so weights
        #   spread across GPUs; otherwise whole model on `self.device` (one card).
        resolved_map: Any = device_map
        n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if resolved_map is None and n_cuda > 1:
            resolved_map = "auto"
            logger.info(
                "[Agent %s] device_map not set; using 'auto' (%d CUDA device(s) visible). "
                "Set model.device_map in config to override (e.g. 'cuda:0' for one full model).",
                agent_id,
                n_cuda,
            )
        if resolved_map is not None:
            load_kwargs["device_map"] = resolved_map
            if max_memory is not None:
                load_kwargs["max_memory"] = max_memory
        else:
            load_kwargs["device_map"] = self.device

        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

        self._model_input_device = getattr(
            self.model, "device", next(self.model.parameters()).device
        )

        hf_dm = getattr(self.model, "hf_device_map", None)
        if hf_dm:
            uniq = sorted({str(v) for v in hf_dm.values()})
            logger.info(
                "[Agent %s] hf_device_map: %d tensors → device(s): %s",
                agent_id,
                len(hf_dm),
                ", ".join(uniq),
            )

        self.hidden_size = self.model.config.hidden_size
        logger.info(
            f"[Agent {agent_id}] Loaded. hidden_size={self.hidden_size}, "
            f"anchor_device={self.device}, model_input_device={self._model_input_device}"
        )

    @property
    def model_input_device(self) -> torch.device:
        """Device for tokenized tensors fed to ``model`` (embedding / first layer)."""
        return self._model_input_device

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------

    def _eos_token_id_set(self) -> Set[int]:
        """Tokenizer / model config may expose eos as int or sequence."""
        ids: Set[int] = set()
        for src in (
            self.tokenizer.eos_token_id,
            getattr(self.model.config, "eos_token_id", None),
        ):
            if src is None:
                continue
            if isinstance(src, (list, tuple)):
                ids.update(int(x) for x in src)
            else:
                ids.add(int(src))
        return ids

    def _hit_max_new_tokens_truncation(
        self, generated_ids_1d: torch.Tensor, max_new_tokens: int
    ) -> bool:
        """True iff generation used the full ``max_new_tokens`` budget without ending on EOS.

        Uses **raw token ids** from ``model.generate`` (the slice after the prompt), not the
        string from ``decode``. ``decode(..., skip_special_tokens=True)`` strips specials from
        the human-readable text only; it does not remove ids from ``generated_ids_1d``, so a
        trailing EOS id (if present in the tensor) is still visible here as ``last``.
        """
        n = int(generated_ids_1d.numel())
        if n == 0 or n < max_new_tokens:
            return False
        last = int(generated_ids_1d[-1].item())
        eos_ids = self._eos_token_id_set()
        if not eos_ids:
            return n >= max_new_tokens
        return last not in eos_ids

    def build_prompt(self, question: str, context: str = "") -> str:
        """Build chat-formatted prompt."""
        messages = [{"role": "system", "content": self.system_prompt}]
        if context:
            messages.append({"role": "user", "content": context})
            messages.append(
                {
                    "role": "assistant",
                    "content": "I've reviewed the prior discussion. Let me continue.",
                }
            )
        messages.append({"role": "user", "content": question})

        # Use chat template if available, else fallback
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = "\n".join(
                f"<|{m['role']}|>\n{m['content']}" for m in messages
            )
            prompt += "\n<|assistant|>\n"
        return prompt

    @torch.no_grad()
    def generate_with_truncation_info(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> tuple[str, bool, dict[str, int]]:
        prompt = self.build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_input_device)
        input_len = inputs["input_ids"].shape[-1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        generated = outputs[0, input_len:]  # LongTensor of new token ids (may end with EOS id)
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        truncated = self._hit_max_new_tokens_truncation(generated, max_new_tokens)
        prompt_tokens = int(input_len)
        completion_tokens = int(generated.numel())
        token_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        return text, truncated, token_usage

    @torch.no_grad()
    def generate(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        text, _, _ = self.generate_with_truncation_info(
            question,
            context,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )
        return text

    @torch.no_grad()
    def generate_batch_with_truncation_info(
        self,
        questions: List[str],
        context: str = "",
        contexts: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[tuple[str, bool, dict[str, int]]]:
        """Generate for a batch of questions in one forward pass (left-pad for decode).

        If ``contexts`` is set, it must match ``questions`` in length; each row uses
        ``build_prompt(questions[i], contexts[i])``. Otherwise all rows share ``context``.
        """
        if not questions:
            return []
        if contexts is not None and len(contexts) != len(questions):
            raise ValueError(
                "contexts must match questions length "
                f"({len(contexts)} != {len(questions)})"
            )
        if len(questions) == 1:
            ctx = contexts[0] if contexts is not None else context
            t, tr, tu = self.generate_with_truncation_info(
                questions[0], ctx, max_new_tokens, temperature, top_p, do_sample
            )
            return [(t, tr, tu)]

        if contexts is not None:
            prompts = [
                self.build_prompt(q, c) for q, c in zip(questions, contexts)
            ]
        else:
            prompts = [self.build_prompt(q, context) for q in questions]
        self.tokenizer.padding_side = "left"
        max_len = getattr(
            self.model.config,
            "max_position_embeddings",
            getattr(self.model.config, "max_sequence_length", 4096),
        ) - max_new_tokens
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            return_attention_mask=True,
        ).to(self._model_input_device)
        self.tokenizer.padding_side = "right"

        attention_mask = inputs["attention_mask"]
        prompt_real_lengths = attention_mask.sum(dim=1)
        input_padded_len = inputs["input_ids"].shape[1]

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        pad_id = self.tokenizer.pad_token_id
        results: List[tuple[str, bool, dict[str, int]]] = []
        for i in range(len(questions)):
            generated_ids = outputs[i, input_padded_len:]
            if pad_id is not None:
                non_pad = generated_ids != pad_id
                if non_pad.any():
                    generated_ids = generated_ids[: non_pad.nonzero()[-1].item() + 1]
                else:
                    generated_ids = generated_ids[:0]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            truncated = self._hit_max_new_tokens_truncation(
                generated_ids, max_new_tokens
            )
            prompt_tokens = int(prompt_real_lengths[i].item())
            completion_tokens = int(generated_ids.numel())
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            results.append((text, truncated, token_usage))
        return results

    @torch.no_grad()
    def generate_batch(
        self,
        questions: List[str],
        context: str = "",
        contexts: Optional[List[str]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[str]:
        return [
            t
            for t, _, _ in self.generate_batch_with_truncation_info(
                questions,
                context=context,
                contexts=contexts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
            )
        ]

    # ------------------------------------------------------------------
    # Hidden-state extraction (used by TFlow as the PG condition input)
    # ------------------------------------------------------------------

    def _find_question_token_range(
        self, question: str, full_prompt: str, input_ids: torch.Tensor
    ) -> tuple[int, int]:
        """Return (start, end) token indices of the *question* inside the full prompt.

        Heuristic: tokenize everything up-to (but excluding) the question text,
        then up-to (and including) the question text.  The difference gives the
        question span.  Falls back to the full sequence on error.
        """
        try:
            idx = full_prompt.rfind(question)
            if idx == -1:
                return 0, int(input_ids.shape[-1])
            prefix = full_prompt[:idx]
            prefix_ids = self.tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"]
            prefix_with_q = full_prompt[:idx + len(question)]
            pwq_ids = self.tokenizer(prefix_with_q, return_tensors="pt", add_special_tokens=True)["input_ids"]
            start = int(prefix_ids.shape[-1])
            end = int(pwq_ids.shape[-1])
            if end <= start:
                return 0, int(input_ids.shape[-1])
            return start, end
        except Exception:
            return 0, int(input_ids.shape[-1])

    @torch.no_grad()
    def extract_hidden_states(
        self, question: str, context: str = ""
    ) -> torch.Tensor:
        """Run forward pass and return last-layer hidden states.

        Returns:
            Tensor of shape (seq_len, hidden_size)
        """
        prompt = self.build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_input_device)

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1].squeeze(0).to(self.device)
        return last_hidden

    @torch.no_grad()
    def extract_hidden_states_all_layers(
        self,
        question: str,
        context: str = "",
        *,
        question_only: bool = False,
    ) -> list[torch.Tensor]:
        """Return hidden states from ALL layers (embedding + decoder layers).

        Args:
            question_only: if True, slice to only question-token positions.

        Returns:
            List of tensors, each (seq_len, hidden_size).  Length = num_layers + 1.
        """
        prompt = self.build_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self._model_input_device)
        input_ids = inputs["input_ids"]

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)

        if question_only:
            start, end = self._find_question_token_range(question, prompt, input_ids)
        else:
            start, end = 0, int(input_ids.shape[-1])

        result: list[torch.Tensor] = []
        for hs in outputs.hidden_states:
            result.append(hs.squeeze(0)[start:end].to(self.device))
        return result

    @torch.no_grad()
    def extract_hidden_states_all_layers_batch(
        self,
        questions: list[str],
        context: str = "",
        *,
        question_only: bool = False,
        max_length: int | None = None,
    ) -> list[list[torch.Tensor]]:
        """Batch version of :meth:`extract_hidden_states_all_layers`.

        Returns:
            outer list = batch, inner list = layers, tensor = (seq_len_i, hidden_size)
        """
        if not questions:
            return []
        prompts = [self.build_prompt(q, context) for q in questions]
        old_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=max_length, return_attention_mask=True,
        ).to(self._model_input_device)
        self.tokenizer.padding_side = old_side

        outputs = self.model(**inputs, output_hidden_states=True, return_dict=True)
        attn_mask = inputs["attention_mask"]
        batch_size = len(questions)

        ranges: list[tuple[int, int]] = []
        for i in range(batch_size):
            if question_only:
                seq_len_i = int(attn_mask[i].sum().item())
                full_prompt_i = prompts[i]
                ids_i = inputs["input_ids"][i, :seq_len_i]
                s, e = self._find_question_token_range(questions[i], full_prompt_i, ids_i)
                ranges.append((s, min(e, seq_len_i)))
            else:
                seq_len_i = int(attn_mask[i].sum().item())
                ranges.append((0, seq_len_i))

        result: list[list[torch.Tensor]] = [[] for _ in range(batch_size)]
        for hs in outputs.hidden_states:
            for i in range(batch_size):
                s, e = ranges[i]
                result[i].append(hs[i, s:e, :].to(self.device))
        return result

    @torch.no_grad()
    def extract_hidden_states_batch(
        self,
        questions: List[str],
        context: str = "",
        *,
        max_length: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Batch forward pass to obtain last-layer hidden states.

        This is the main throughput lever for TFlow evaluation when generate_bs > 1.

        Args:
            questions: list of questions (batch).
            context: optional shared context.
            max_length: optional tokenizer truncation length (prompt length).

        Returns:
            last_hidden: (batch, seq_len, hidden_size) padded on the right
            attention_mask: (batch, seq_len) with 1 for real tokens, 0 for padding
        """
        if not questions:
            empty = torch.empty(0, 0, self.hidden_size, device=self.device)
            empty_mask = torch.empty(0, 0, device=self.device, dtype=torch.long)
            return empty, empty_mask

        prompts = [self.build_prompt(q, context) for q in questions]
        old_side = getattr(self.tokenizer, "padding_side", "right")
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
        ).to(self._model_input_device)
        self.tokenizer.padding_side = old_side

        outputs = self.model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
        last_hidden = outputs.hidden_states[-1].to(self.device)  # (batch, seq_len, hidden_size)
        attention_mask = inputs["attention_mask"].to(self.device)  # (batch, seq_len)
        return last_hidden, attention_mask

    @torch.no_grad()
    def generate_with_hidden_states(
        self,
        question: str,
        context: str = "",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> tuple[str, torch.Tensor]:
        """Generate text AND return hidden states of the full sequence."""
        # First generate text
        text = self.generate(
            question, context, max_new_tokens, temperature, do_sample=do_sample
        )
        # Then do a forward pass on full (prompt + generated) to get hidden states
        full_prompt = self.build_prompt(question, context) + text
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self._model_input_device)

        outputs = self.model(
            **inputs, output_hidden_states=True, return_dict=True
        )
        last_hidden = outputs.hidden_states[-1].squeeze(0).to(self.device)
        return text, last_hidden