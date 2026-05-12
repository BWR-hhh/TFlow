#!/usr/bin/env python3
"""Unified evaluation entry point for TFlow (Thought Flow).

Methods (selected via ``--method``):

    baseline : single-agent inference.
    textmas  : text-space multi-agent baseline (sender agents A and B run
               in parallel; executor C reads their concatenated outputs).
    tflow    : weight-space multi-agent inference; sender agents A and B
               condition a ``ParameterGenerator`` that emits LoRA factor
               pairs ``(A, B)``, which are convex-combined and applied to
               receiver agent C via per-sample forward hooks.

Supported benchmarks (``--dataset``):

    gsm8k, mbppplus (alias ``mbpp``), humanevalplus (alias ``humaneval``),
    minerva_math, mmlu.

Examples::

    python main.py --method baseline --dataset gsm8k
    python main.py --method textmas  --dataset gsm8k --generate_bs 8
    python main.py --method tflow    --dataset humanevalplus \\
        --checkpoint /path/to/tflow_best.pt --generate_bs 8

All configurations live in :mod:`src.configs` (no external YAML files).
"""

from __future__ import annotations

import os
import sys
import json
import copy
import argparse
import logging
from datetime import datetime

from src.configs import AVAILABLE_METHODS, get_config
from src.utils.helpers import set_seed, setup_logging
from src.methods import build_method
from src.data.datasets import effective_hf_cache_dir, load_dataset_for_eval
from src.evaluation.evaluator import Evaluator

logger = logging.getLogger(__name__)


def build_eval_run_metadata(
    config: dict,
    args: argparse.Namespace,
    output_dir: str,
) -> dict:
    """Snapshot of the resolved config and CLI for the result JSON."""
    cfg = copy.deepcopy(config)
    cfg.pop("_mode", None)

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_dir": output_dir,
        "cwd": os.getcwd(),
        "command_line": " ".join(sys.argv),
        "seed": cfg.get("seed"),
        "method": cfg.get("method"),
        "cli": {
            "method": args.method,
            "output_dir": args.output_dir,
            "max_samples": args.max_samples,
            "dataset": args.dataset,
            "generate_bs": args.generate_bs,
            "max_new_tokens": args.max_new_tokens,
            "checkpoint": args.checkpoint,
        },
        "model": cfg.get("model", {}),
        "data": cfg.get("data", {}),
        "evaluation": cfg.get("evaluation", {}),
        "logging": {"output_dir": cfg.get("logging", {}).get("output_dir")},
        "method_config": cfg.get("method_config", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TFlow (Thought Flow) inference-only evaluator")
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=AVAILABLE_METHODS,
        help="Inference method: 'baseline', 'textmas', or 'tflow'. "
             "Configurations are defined in src/configs.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override logging.output_dir (default: ./outputs under the current "
             "working directory).",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Cap the number of evaluation samples (-1 = full split).",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Override data.dataset; one of "
             "gsm8k, mbppplus (mbpp), humanevalplus (humaneval), minerva_math, mmlu.",
    )
    parser.add_argument(
        "--generate_bs",
        type=int,
        default=None,
        help="Override evaluation.generate_bs: per-step batch size for generation "
             "(>1 enables true batched inference for all three methods).",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override model.max_new_tokens; takes precedence over the "
             "per-dataset max_new_tokens_by_dataset table.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the TFlow evaluation checkpoint (only valid for method=tflow).",
    )
    return parser.parse_args()


def run_eval(config: dict) -> dict:
    """Build the method, load the requested dataset, and run evaluation."""
    method = build_method(config)
    method.setup()

    data_cfg = config["data"]
    dataset = load_dataset_for_eval(
        name=data_cfg["dataset"],
        split=data_cfg.get("split", "test"),
        max_samples=data_cfg.get("max_samples", -1),
        cache_dir=effective_hf_cache_dir(data_cfg.get("cache_dir")),
    )

    evaluator = Evaluator(config)
    return evaluator.evaluate(method, dataset)


def main() -> None:
    args = parse_args()
    config = get_config(args.method)

    if args.output_dir:
        config.setdefault("logging", {})["output_dir"] = args.output_dir
    if args.max_samples is not None:
        config.setdefault("data", {})["max_samples"] = args.max_samples
    if args.dataset is not None:
        config.setdefault("data", {})["dataset"] = args.dataset
    if args.generate_bs is not None:
        config.setdefault("evaluation", {})["generate_bs"] = args.generate_bs
    if args.max_new_tokens is not None:
        model_cfg = config.setdefault("model", {})
        model_cfg["max_new_tokens"] = args.max_new_tokens
        model_cfg.pop("max_new_tokens_by_dataset", None)

    if args.checkpoint is not None:
        if config.get("method") != "tflow":
            raise ValueError("--checkpoint is only valid when method == 'tflow'.")
        mc = config.setdefault("method_config", {})
        train_cfg = mc.setdefault("training", {})
        train_cfg["eval_checkpoint"] = args.checkpoint

    config["_mode"] = "eval"

    base_output_dir = config.get("logging", {}).get("output_dir") or os.path.abspath(
        "outputs"
    )
    method_name = config.get("method", "unknown")
    dataset_name = config.get("data", {}).get("dataset", "unknown")
    run_subdir = "{}_{}_{}_eval".format(
        datetime.now().strftime("%Y%m%d_%H%M%S"),
        method_name,
        dataset_name,
    )
    output_dir = os.path.join(base_output_dir, run_subdir)
    os.makedirs(output_dir, exist_ok=True)

    setup_logging(output_dir)
    if args.dataset is not None:
        logger.info(f"Dataset overridden by CLI: {args.dataset}")
    set_seed(config.get("seed", 42))

    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    logger.info(f"  TFlow (inference) | Method: {method_name} | Dataset: {dataset_name}")
    logger.info("=" * 60)

    results = run_eval(config)
    results["eval_run"] = build_eval_run_metadata(config, args, output_dir)
    out_path = os.path.join(output_dir, f"{method_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
