#!/usr/bin/env bash
# Convenience driver: evaluate a single inference method across all five
# benchmarks.  Useful as a quick smoke test (default 50 samples per
# dataset) or for full reproduction (set MAX_SAMPLES=-1).
#
# Usage:
#   bash scripts/run_eval.sh                                 # baseline, 50 samples each
#   bash scripts/run_eval.sh textmas                         # textmas
#   bash scripts/run_eval.sh tflow                           # tflow with default checkpoint search
#   bash scripts/run_eval.sh tflow /path/to/ckpt.pt          # tflow with explicit checkpoint
#   MAX_SAMPLES=-1 bash scripts/run_eval.sh                  # full evaluation
#   GENERATE_BS=16 bash scripts/run_eval.sh                  # increase batch size

set -euo pipefail

METHOD=${1:-baseline}
CHECKPOINT=${2:-}
MAX_SAMPLES=${MAX_SAMPLES:-50}
GENERATE_BS=${GENERATE_BS:-8}

DATASETS=(
  gsm8k
  mbpp
  humaneval
  minerva_math
  mmlu
)

EXTRA_ARGS=()
if [[ -n "${CHECKPOINT}" ]]; then
  EXTRA_ARGS+=(--checkpoint "${CHECKPOINT}")
fi

echo "[run_eval] method=${METHOD} max_samples=${MAX_SAMPLES} generate_bs=${GENERATE_BS}"
echo "[run_eval] datasets=${DATASETS[*]}"

for ds in "${DATASETS[@]}"; do
  echo "============================================================"
  echo "[run_eval] dataset=${ds}"
  echo "============================================================"
  python main.py \
    --method "${METHOD}" \
    --dataset "${ds}" \
    --max_samples "${MAX_SAMPLES}" \
    --generate_bs "${GENERATE_BS}" \
    "${EXTRA_ARGS[@]}"
done
