"""Infer ParameterGenerator ``pg_mapping`` from a HuggingFace causal LM (e.g. Qwen3)."""

from __future__ import annotations

import logging
from typing import Dict, Optional, Sequence

import torch.nn as nn

logger = logging.getLogger(__name__)


def _linear_lora_dims(mod: nn.Module) -> tuple[int, int] | None:
    """Return (lora_A_dim, lora_B_dim) for ΔW=B@A matching ``Linear.weight`` (out, in)."""
    w = getattr(mod, "weight", None)
    if w is None or not isinstance(w, nn.Parameter):
        return None
    if w.dim() != 2:
        return None
    out_f, in_f = int(w.shape[0]), int(w.shape[1])
    return in_f, out_f


def _apply_include_substrings_filter(
    out: Dict[str, Dict[str, int]],
    include_substrings: Optional[Sequence[str]],
) -> Dict[str, Dict[str, int]]:
    if not include_substrings:
        return out
    needles = [str(s) for s in include_substrings if s]
    if not needles:
        return out
    filtered = {k: v for k, v in out.items() if any(n in k for n in needles)}
    if not filtered:
        raise ValueError(
            "pg_mapping_include_substrings 过滤后为空；当前可用键: "
            + ", ".join(sorted(out.keys()))
        )
    return filtered


def infer_pg_mapping_all_linears_in_layer(layer0: nn.Module) -> Dict[str, Dict[str, int]]:
    """为 ``layer0``（单层 decoder block）下**每一个** ``nn.Linear``（二维权重）建一条 ``pg_mapping``.

    键为相对 ``model.layers[i].`` 的路径（与 ``named_modules()`` 一致，如 ``self_attn.q_proj``）。
    MoE 等结构会为每个 expert 的每个 Linear 单独建项，PG token 量会显著变大。
    """
    out: Dict[str, Dict[str, int]] = {}
    for name, mod in layer0.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        dims = _linear_lora_dims(mod)
        if dims is None:
            continue
        in_f, out_f = dims
        key = name
        if not key:
            continue
        out[key] = {"lora_A_dim": in_f, "lora_B_dim": out_f}
    out = dict(sorted(out.items()))
    if not out:
        raise ValueError(
            "infer_pg_mapping_all_linears_in_layer: no nn.Linear with 2-D weight in layer 0."
        )
    return out


def infer_pg_mapping_from_causal_lm(
    model: nn.Module,
    *,
    include_substrings: Optional[Sequence[str]] = None,
    all_linears: bool = False,
) -> Dict[str, Dict[str, int]]:
    """Build dense ``pg_mapping`` keys (no layer index) from the first decoder layer.

    If ``all_linears`` is True, every ``nn.Linear`` under layer 0 is included (see
    :func:`infer_pg_mapping_all_linears_in_layer`). MoE / 大量子模块时显存与 PG 算力开销会很高。

    Otherwise, resolution order:

    1. **Fused-style**: ``self_attn.qkv_proj`` + ``self_attn.o_proj`` and/or
       ``mlp.shared_mlp.{gate_and_up_proj,down_proj}`` when those modules exist.
    2. **Split attention** (Qwen3/Llama): ``self_attn.{q,k,v,o}_proj``.
    3. **Split MLP**: ``mlp.{gate_proj,up_proj,down_proj}`` if ``shared_mlp`` is absent.

    Skips missing modules. Fused ``qkv_proj`` and separate ``q_proj``/… are mutually exclusive
    (prefer fused when present).
    """
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise ValueError(
            "Cannot infer pg_mapping: expected model.model.layers (HF CausalLM)."
        )
    layer0 = layers[0]

    if all_linears:
        out = infer_pg_mapping_all_linears_in_layer(layer0)
        n = len(out)
        if n > 48:
            logger.warning(
                "[param_gen] pg_mapping_all_linears: %d Linear targets — PG token count and VRAM "
                "can be very large; consider pg_mapping_all_linears: false with a narrow "
                "pg_mapping_include_substrings.",
                n,
            )
        out = _apply_include_substrings_filter(out, include_substrings)
        logger.info(
            "[param_gen] Inferred pg_mapping (all_linears, %d targets): %s",
            len(out),
            ", ".join(sorted(out.keys())),
        )
        return out

    out = {}

    attn = getattr(layer0, "self_attn", None) or getattr(layer0, "attention", None)
    if attn is not None:
        qkv = getattr(attn, "qkv_proj", None)
        if qkv is not None:
            dims = _linear_lora_dims(qkv)
            if dims is not None:
                in_f, out_f = dims
                out["self_attn.qkv_proj"] = {"lora_A_dim": in_f, "lora_B_dim": out_f}
        else:
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(attn, name, None)
                if mod is None:
                    continue
                dims = _linear_lora_dims(mod)
                if dims is None:
                    continue
                in_f, out_f = dims
                key = f"self_attn.{name}"
                out[key] = {"lora_A_dim": in_f, "lora_B_dim": out_f}

        o_mod = getattr(attn, "o_proj", None)
        if o_mod is not None:
            dims = _linear_lora_dims(o_mod)
            if dims is not None:
                in_f, out_f = dims
                out["self_attn.o_proj"] = {"lora_A_dim": in_f, "lora_B_dim": out_f}

    mlp = getattr(layer0, "mlp", None) or getattr(layer0, "feed_forward", None)
    if mlp is not None:
        shared = getattr(mlp, "shared_mlp", None)
        if shared is not None:
            gau = getattr(shared, "gate_and_up_proj", None)
            if gau is not None:
                dims = _linear_lora_dims(gau)
                if dims is not None:
                    in_f, out_f = dims
                    out["mlp.shared_mlp.gate_and_up_proj"] = {
                        "lora_A_dim": in_f,
                        "lora_B_dim": out_f,
                    }
            down = getattr(shared, "down_proj", None)
            if down is not None:
                dims = _linear_lora_dims(down)
                if dims is not None:
                    in_f, out_f = dims
                    out["mlp.shared_mlp.down_proj"] = {
                        "lora_A_dim": in_f,
                        "lora_B_dim": out_f,
                    }
        else:
            for name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(mlp, name, None)
                if mod is None:
                    continue
                dims = _linear_lora_dims(mod)
                if dims is None:
                    continue
                in_f, out_f = dims
                out[f"mlp.{name}"] = {"lora_A_dim": in_f, "lora_B_dim": out_f}

    if not out:
        raise ValueError(
            "Cannot infer pg_mapping: no supported Linear modules in layer 0 "
            "(expected self_attn q/k/v/o_proj and/or mlp gate/up/down_proj)."
        )

    out = _apply_include_substrings_filter(out, include_substrings)

    logger.info(
        "[param_gen] Inferred pg_mapping from backbone (%d targets): %s",
        len(out),
        ", ".join(sorted(out.keys())),
    )
    return out
