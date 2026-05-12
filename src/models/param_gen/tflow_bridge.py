"""ParameterGenerator outputs → runtime LoRA factors (A, B) + scaling (α/rank)."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import torch

logger = logging.getLogger(__name__)

# PG 原始 lora_A / lora_B 先乘该系数再作为 LoRA 参数；前向为 x @ A^T @ B^T，再乘 scale（α/rank）。
PG_OUTPUT_SCALE = 0.01

LoRAFactorTuple = Tuple[torch.Tensor, torch.Tensor, float]


def _pg_scale(pg_state: dict) -> float:
    s = pg_state.get("scale")
    if s is None:
        return 1.0
    if isinstance(s, torch.Tensor):
        return float(s.detach().float().item())
    return float(s)


def _squeeze_leading_unit_batch(t: torch.Tensor) -> torch.Tensor:
    if t.dim() >= 1 and t.shape[0] == 1:
        return t.squeeze(0)
    return t


def pg_state_dict_to_lora_factors(
    pg_state: dict,
    pg_output_scale: float = PG_OUTPUT_SCALE,
) -> Dict[str, LoRAFactorTuple]:
    """Map PG output (lora_A / lora_B per module) to ``{full_param_name.weight: (A, B, scaling)}``.

    - ``A`` = PG 的 lora_A × ``pg_output_scale`` (shape ``(rank, in_features)``).
    - ``B`` = PG 的 lora_B × ``pg_output_scale`` (shape ``(out_features, rank)``).
    - ``scaling`` = ``pg_state['scale']``（与 Tokenizer2DBatchedLoRA 一致，一般为 α/rank）。

    运行时增量为 ``(x @ A^T @ B^T) * scaling``（与物化 ``ΔW = (B @ A) * scaling`` 再加到 ``F.linear`` 等价）。

    Skips expert-shaped (4D+) tensors (e.g. MoE) that do not match a single Linear.
    """
    scale = _pg_scale(pg_state)
    bases: set[str] = set()
    for k in pg_state:
        if k == "scale":
            continue
        if k.endswith(".lora_A"):
            bases.add(k[: -len(".lora_A")])
        elif k.endswith(".lora_B"):
            bases.add(k[: -len(".lora_B")])

    out: Dict[str, LoRAFactorTuple] = {}
    for base in sorted(bases):
        ka, kb = f"{base}.lora_A", f"{base}.lora_B"
        if ka not in pg_state or kb not in pg_state:
            continue
        A = _squeeze_leading_unit_batch(pg_state[ka]) * pg_output_scale
        B = _squeeze_leading_unit_batch(pg_state[kb]) * pg_output_scale
        if A.dim() != 2 or B.dim() != 2:
            logger.warning(
                "[param_gen] Skip LoRA pair for %s: A.dim=%s B.dim=%s (MoE / batched expert?)",
                base,
                A.dim(),
                B.dim(),
            )
            continue
        out[f"{base}.weight"] = (A, B, scale)
    return out


def pg_state_dict_to_batched_lora_factors(
    pg_state: dict,
    pg_output_scale: float = PG_OUTPUT_SCALE,
) -> Dict[str, LoRAFactorTuple]:
    """Like :func:`pg_state_dict_to_lora_factors` but preserves the batch dimension.

    Returns ``{key.weight: (A, B, scale)}`` where ``A`` is ``(B, rank, in)``
    and ``B`` is ``(B, out, rank)``.  Used by the hook-based batched generation
    path where each sample in the batch has its own LoRA factors.
    """
    scale = _pg_scale(pg_state)
    bases: set[str] = set()
    for k in pg_state:
        if k == "scale":
            continue
        if k.endswith(".lora_A"):
            bases.add(k[: -len(".lora_A")])
        elif k.endswith(".lora_B"):
            bases.add(k[: -len(".lora_B")])

    out: Dict[str, LoRAFactorTuple] = {}
    for base in sorted(bases):
        ka, kb = f"{base}.lora_A", f"{base}.lora_B"
        if ka not in pg_state or kb not in pg_state:
            continue
        A = pg_state[ka] * pg_output_scale
        B = pg_state[kb] * pg_output_scale
        if A.dim() == 2:
            A = A.unsqueeze(0)
        if B.dim() == 2:
            B = B.unsqueeze(0)
        if A.dim() != 3 or B.dim() != 3:
            logger.warning(
                "[param_gen] batched: skip %s A.dim=%s B.dim=%s",
                base,
                A.dim(),
                B.dim(),
            )
            continue
        out[f"{base}.weight"] = (A, B, scale)
    return out


def pg_state_to_per_sample_lora_factors(
    pg_state: dict,
    pg_output_scale: float = PG_OUTPUT_SCALE,
) -> List[Dict[str, LoRAFactorTuple]]:
    """When PG was run with batch ``B`` on condition, split into ``B`` factor dicts."""
    scale = _pg_scale(pg_state)
    bases: set[str] = set()
    for k in pg_state:
        if k == "scale":
            continue
        if k.endswith(".lora_A"):
            bases.add(k[: -len(".lora_A")])
        elif k.endswith(".lora_B"):
            bases.add(k[: -len(".lora_B")])

    batch_size = 1
    for base in bases:
        A = pg_state.get(f"{base}.lora_A")
        if A is None:
            continue
        if A.dim() == 3:
            batch_size = int(A.shape[0])
        break

    out_list: List[Dict[str, LoRAFactorTuple]] = [{} for _ in range(batch_size)]

    for base in sorted(bases):
        ka, kb = f"{base}.lora_A", f"{base}.lora_B"
        if ka not in pg_state or kb not in pg_state:
            continue
        A = pg_state[ka]
        Bt = pg_state[kb]
        if A.dim() == 2 and Bt.dim() == 2:
            a = A * pg_output_scale
            b = Bt * pg_output_scale
            for bidx in range(batch_size):
                out_list[bidx][f"{base}.weight"] = (a, b, scale)
        elif A.dim() == 3 and Bt.dim() == 3:
            for bidx in range(batch_size):
                out_list[bidx][f"{base}.weight"] = (
                    A[bidx] * pg_output_scale,
                    Bt[bidx] * pg_output_scale,
                    scale,
                )
        else:
            logger.warning(
                "[param_gen] Skip %s: unexpected A.dim=%s B.dim=%s",
                base,
                A.dim(),
                Bt.dim(),
            )

    return out_list


def fuse_lora_factors(
    factor_dicts: List[Dict[str, LoRAFactorTuple]],
    weights: List[float],
) -> Dict[str, LoRAFactorTuple]:
    """Convex combine LoRA factor dicts: same key -> (sum_i w_i A_i, sum_i w_i B_i, scaling).

    .. warning::

        Pre-fusing A and B separately introduces cross-term error:
        ``(Σ w_i B_i) @ (Σ w_i A_i) ≠ Σ w_i (B_i @ A_i)``.
        Prefer :func:`build_weighted_lora_entries` +
        :func:`~src.models.lora.patch_linear_multi_lora_factors` for correct fusion.
    """
    if not factor_dicts:
        return {}
    if len(factor_dicts) != len(weights):
        raise ValueError(
            f"fuse_lora_factors: len(factor_dicts)={len(factor_dicts)} != len(weights)={len(weights)}"
        )
    keys = set(factor_dicts[0].keys())
    for d in factor_dicts[1:]:
        keys &= set(d.keys())
    fused: Dict[str, LoRAFactorTuple] = {}
    for k in sorted(keys):
        sc0 = factor_dicts[0][k][2]
        A_acc = None
        B_acc = None
        for d, w in zip(factor_dicts, weights):
            A_i, B_i, sc_i = d[k]
            if abs(float(sc_i) - float(sc0)) > 1e-5:
                logger.warning(
                    "[param_gen] fuse_lora_factors: scaling mismatch for %s (%s vs %s), using first.",
                    k,
                    sc_i,
                    sc0,
                )
            t_a = A_i * w
            t_b = B_i * w
            A_acc = t_a if A_acc is None else A_acc + t_a
            B_acc = t_b if B_acc is None else B_acc + t_b
        assert A_acc is not None and B_acc is not None
        fused[k] = (A_acc, B_acc, float(sc0))
    return fused


# -- Weighted multi-sender entries (cross-term–free) ----------------------

WeightedLoRAEntry = Tuple[torch.Tensor, torch.Tensor, float, Any]


def build_weighted_lora_entries(
    factor_dicts: List[Dict[str, LoRAFactorTuple]],
    weights: List[Any],
) -> Dict[str, List[WeightedLoRAEntry]]:
    """Group per-sender LoRA factors by parameter name with per-sender weights.

    Unlike :func:`fuse_lora_factors` which pre-combines A and B matrices
    (introducing cross-term errors from ``(Σw·A)(Σw·B) ≠ Σw·(B@A)``), this
    preserves individual sender contributions so that the patched forward
    correctly computes ``Σ w_i · (x @ A_i^T @ B_i^T) · scale_i``.

    ``weights`` can be plain floats **or** scalar :class:`torch.Tensor` values
    (to allow gradient flow through a learned fusion gate).
    """
    if len(factor_dicts) != len(weights):
        raise ValueError(
            f"build_weighted_lora_entries: len(factor_dicts)={len(factor_dicts)} "
            f"!= len(weights)={len(weights)}"
        )
    all_keys: set[str] = set()
    for fd in factor_dicts:
        all_keys.update(fd.keys())

    grouped: Dict[str, List[WeightedLoRAEntry]] = {}
    for k in sorted(all_keys):
        entries: List[WeightedLoRAEntry] = []
        for fd, w in zip(factor_dicts, weights):
            if k in fd:
                A, B, scale = fd[k]
                entries.append((A, B, scale, w))
        if entries:
            grouped[k] = entries
    return grouped
