"""LoRA runtime utilities for the TFlow (ParameterGenerator) inference path.

The utilities apply LoRA factor pairs ``(A, B)`` produced by the
``ParameterGenerator`` to selected ``nn.Linear`` modules of the receiver
backbone.
Two application paths are provided:

- :func:`patch_linear_multi_lora_factors` /
  :func:`restore_forward_patches` — single-sample forward patches for
  ``solve()``;
- :func:`apply_batched_multi_lora_hooks` /
  :func:`remove_batched_multi_lora_hooks` — batched per-sample forward
  hooks for ``solve_batch()``.

Both paths preserve cross-term correctness for multi-sender fusion by
applying each sender's contribution separately and summing.
"""

from __future__ import annotations

import logging
from typing import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ======================================================================
# Single-sample: factor-based forward patch  (used by ``solve``)
# ======================================================================


def patch_linear_multi_lora_factors(
    model: nn.Module,
    weighted_entries: dict[str, list[tuple[torch.Tensor, torch.Tensor, float, "float | torch.Tensor"]]],
) -> list[tuple[nn.Module, Callable[..., torch.Tensor]]]:
    """Patch target ``nn.Linear`` modules with weighted multi-sender LoRA.

    Each patched Linear's forward becomes
    ``y + Σ w_i · (x @ A_i^T @ B_i^T) · scale_i`` so cross-terms between
    different senders' factors are avoided.

    ``weighted_entries`` maps full parameter names ending in ``.weight`` to a
    list of ``(A_i, B_i, scale_i, weight_i)`` tuples — one per sender.
    ``weight_i`` may be a plain float **or** a scalar :class:`torch.Tensor`.

    Returns:
        List of ``(module, original_forward)`` for :func:`restore_forward_patches`.
    """
    patches: list[tuple[nn.Module, Callable[..., torch.Tensor]]] = []
    for pname, entries in weighted_entries.items():
        if not pname.endswith(".weight"):
            logger.warning("[lora] patch_linear_multi_lora_factors: skip key %r", pname)
            continue
        tokens = pname.split(".")
        if len(tokens) < 2 or tokens[-1] != "weight":
            continue
        module_path = ".".join(tokens[:-1])
        mod: nn.Module = model
        try:
            for tok in module_path.split("."):
                mod = getattr(mod, tok)
        except AttributeError:
            logger.warning("[lora] Cannot resolve module for %s", pname)
            continue

        orig_forward = mod.forward

        def _make_patched(
            orig_fn: Callable[..., torch.Tensor],
            lora_entries: list[tuple[torch.Tensor, torch.Tensor, float, "float | torch.Tensor"]],
        ) -> Callable[..., torch.Tensor]:
            def patched(x: torch.Tensor) -> torch.Tensor:
                out = orig_fn(x)
                for a, b, sc, w in lora_entries:
                    aa = a.to(device=x.device, dtype=x.dtype)
                    bb = b.to(device=x.device, dtype=x.dtype)
                    lora_out = (x @ aa.T @ bb.T) * sc
                    if isinstance(w, torch.Tensor):
                        lora_out = lora_out * w.to(device=x.device, dtype=x.dtype)
                    else:
                        lora_out = lora_out * float(w)
                    out = out + lora_out
                return out

            return patched

        mod.forward = _make_patched(orig_forward, entries)  # type: ignore[method-assign]
        patches.append((mod, orig_forward))

    logger.debug("patch_linear_multi_lora_factors: patched %d Linear modules.", len(patches))
    return patches


def restore_forward_patches(
    patches: list[tuple[nn.Module, Callable[..., torch.Tensor]]],
) -> None:
    """Restore ``forward`` from :func:`patch_linear_multi_lora_factors`."""
    for mod, orig in patches:
        mod.forward = orig  # type: ignore[method-assign]


# ======================================================================
# Batched per-sample LoRA hooks  (used by ``solve_batch``)
# ======================================================================

_MULTI_LORA_HOOKS: dict[int, list] = {}


def apply_batched_multi_lora_hooks(
    model: nn.Module,
    batched_entries: dict[str, list[tuple[torch.Tensor, torch.Tensor, float, "float | torch.Tensor"]]],
) -> None:
    """Register forward hooks for truly batched per-sample multi-sender LoRA.

    Each entry's ``A`` tensor has shape ``(B, rank, in_features)`` and ``B``
    tensor has shape ``(B, out_features, rank)``.  Weights can be scalar floats
    or ``(B,)`` tensors for per-sample gating.

    Handles batch-size expansion (beam search) transparently.
    """
    model_id = id(model)
    if model_id in _MULTI_LORA_HOOKS:
        remove_batched_multi_lora_hooks(model)
    _MULTI_LORA_HOOKS[model_id] = []

    for pname, entries in batched_entries.items():
        if not pname.endswith(".weight"):
            continue
        tokens = pname.split(".")
        if len(tokens) < 2 or tokens[-1] != "weight":
            continue
        module_path = ".".join(tokens[:-1])
        mod: nn.Module = model
        try:
            for tok in module_path.split("."):
                mod = getattr(mod, tok)
        except AttributeError:
            logger.warning("[lora] batched hooks: cannot resolve %s", pname)
            continue

        def _make_hook(
            lora_entries: list[tuple[torch.Tensor, torch.Tensor, float, "float | torch.Tensor"]],
        ):
            def hook(_module, _input, output):
                x = _input[0]  # (B_cur, seq, in_features)
                cur_b = x.shape[0]
                delta = torch.zeros_like(output)
                for a_bat, b_bat, sc, w in lora_entries:
                    aa = a_bat.to(device=x.device, dtype=x.dtype)
                    bb = b_bat.to(device=x.device, dtype=x.dtype)
                    orig_b = aa.shape[0]
                    if cur_b != orig_b and orig_b > 0 and cur_b % orig_b == 0:
                        rn = cur_b // orig_b
                        aa = aa.repeat(rn, 1, 1)
                        bb = bb.repeat(rn, 1, 1)
                    h = torch.bmm(x, aa.transpose(1, 2))
                    lora_out = torch.bmm(h, bb.transpose(1, 2)) * sc
                    if isinstance(w, torch.Tensor):
                        w_val = w.to(device=x.device, dtype=x.dtype)
                        if w_val.dim() == 0:
                            lora_out = lora_out * w_val
                        else:
                            if w_val.shape[0] != cur_b and cur_b % w_val.shape[0] == 0:
                                w_val = w_val.repeat(cur_b // w_val.shape[0])
                            lora_out = lora_out * w_val.view(-1, 1, 1)
                    else:
                        lora_out = lora_out * float(w)
                    delta = delta + lora_out
                return output + delta

            return hook

        handle = mod.register_forward_hook(_make_hook(entries))
        _MULTI_LORA_HOOKS[model_id].append(handle)

    logger.debug(
        "apply_batched_multi_lora_hooks: %d hooks on model %d.",
        len(_MULTI_LORA_HOOKS.get(model_id, [])),
        model_id,
    )


def remove_batched_multi_lora_hooks(model: nn.Module) -> None:
    """Remove all batched multi-sender LoRA hooks from *model*."""
    model_id = id(model)
    hooks = _MULTI_LORA_HOOKS.pop(model_id, [])
    for h in hooks:
        h.remove()
    if hooks:
        logger.debug(
            "Removed %d batched multi-LoRA hooks from model %d.", len(hooks), model_id
        )
