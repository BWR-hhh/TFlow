#!/usr/bin/env python3
"""Run TFlow on a user-provided single sample."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.configs import AVAILABLE_METHODS, get_config
from src.data.datasets import resolve_dataset_name
from src.methods import build_method
from src.utils.helpers import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a single user-provided sample with TFlow, baseline, or TextMAS."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="tflow",
        choices=AVAILABLE_METHODS,
        help="Inference method to run.",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question/sample text. If omitted, use --input_file or stdin.",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to a text file containing the question/sample.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help=(
            "Prompt style to use: gsm8k, minerva_math, mmlu, humaneval, or mbpp. "
            "This does not load a benchmark dataset."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the TFlow checkpoint; only valid with --method tflow.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Override generation length for this sample.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path.",
    )
    return parser.parse_args()


def read_question(args: argparse.Namespace) -> str:
    if args.question is not None:
        text = args.question
    elif args.input_file is not None:
        text = Path(args.input_file).read_text(encoding="utf-8")
    else:
        text = sys.stdin.read()

    question = text.strip()
    if not question:
        raise ValueError("No input sample provided. Use --question, --input_file, or stdin.")
    return question


def build_config(args: argparse.Namespace) -> dict:
    config = get_config(args.method)
    dataset = resolve_dataset_name(args.dataset)
    config.setdefault("data", {})["dataset"] = dataset
    config.setdefault("evaluation", {})["generate_bs"] = 1

    if args.max_new_tokens is not None:
        model_cfg = config.setdefault("model", {})
        model_cfg["max_new_tokens"] = args.max_new_tokens
        model_cfg.pop("max_new_tokens_by_dataset", None)

    if args.checkpoint is not None:
        if args.method != "tflow":
            raise ValueError("--checkpoint is only valid when --method tflow.")
        train_cfg = config.setdefault("method_config", {}).setdefault("training", {})
        train_cfg["eval_checkpoint"] = args.checkpoint

    config["_mode"] = "inference"
    return config


def main() -> None:
    args = parse_args()
    question = read_question(args)
    config = build_config(args)

    set_seed(config.get("seed", 42))
    method = build_method(config)
    method.setup()
    result = method.solve(question)

    payload = {
        "method": args.method,
        "dataset_prompt": config.get("data", {}).get("dataset"),
        "question": question,
        "answer": result.get("answer", ""),
        "metadata": result.get("metadata", {}),
    }

    print(payload["answer"])

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
