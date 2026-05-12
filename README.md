# TFlow: Good Agentic Friends Do Not Just Give Verbal Advice

This repository contains the canonical release code for
**Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update
Your Weights**.

TFlow (Thought Flow) evaluates weight-space inter-agent communication:
a `ParameterGenerator` produces instance-conditioned LoRA updates that
let one agent update another agent's weights during inference.

> Paper link and BibTeX will be added after the public arXiv record is
> available.

## Requirements

To install requirements:

```setup
conda create -n tflow python=3.10 -y
conda activate tflow

pip install -r requirements.txt
pip install -e .
```

The five evaluation benchmarks (GSM8K, MBPP+, HumanEval+, Minerva
Math, MMLU) are downloaded automatically through the Hugging Face
`datasets` library on first use; set `HF_HOME` to control the cache
location.

## Training

Training code is not included in this inference-only release. The
released checkpoint at `checkpoints/tflow/tflow_best.pt` was obtained
with the procedure described in the paper (Qwen3-4B backbone,
`ParameterGenerator` trained for 32k samples on the mixed reasoning
corpus).

```train
# See the paper for training details and hyper-parameters.
```

## Evaluation

To evaluate the released model on the five benchmarks, run:

```eval
python main.py --method tflow --dataset gsm8k        --generate_bs 4
python main.py --method tflow --dataset minerva_math --generate_bs 4
python main.py --method tflow --dataset mmlu         --generate_bs 4
python main.py --method tflow --dataset humaneval    --generate_bs 4
python main.py --method tflow --dataset mbpp         --generate_bs 4
```

You can also run all five datasets through the convenience script:

```eval
MAX_SAMPLES=-1 GENERATE_BS=4 bash scripts/run_eval.sh tflow
```

Without `MAX_SAMPLES=-1`, `scripts/run_eval.sh` defaults to a 50-sample
smoke test per dataset.

### Single-sample Inference

For a quick single-sample inference run, pass a sample directly:

```eval
python infer.py --method tflow --question "Janet has 3 apples and buys 5 more. How many apples does she have?"
```

You can also read the sample from a text file or stdin:

```eval
python infer.py --method tflow --input_file sample.txt
echo "Janet has 3 apples and buys 5 more. How many apples does she have?" | python infer.py --method tflow
```

Use `--dataset` to choose the prompt style without loading a benchmark
dataset:

```eval
python infer.py --method tflow --dataset mmlu --question "Question: ...\nA. ...\nB. ...\nC. ...\nD. ..."
python infer.py --method baseline --question "Solve 17 * 23."
python infer.py --method textmas  --question "Solve 17 * 23."
```

Pass `--output outputs/single_sample.json` to save the question,
prediction, and inference metadata as JSON.

To reproduce the two comparison methods, replace `--method tflow` with
`--method baseline` (single agent) or `--method textmas` (text-channel
multi-agent). Each run writes `run.log` and `{method}_results.json`
under `./outputs/`.

## Pre-trained Models

You can download pretrained models here:

- [TFlow ParameterGenerator (Qwen3-4B backbone)](checkpoints/tflow/tflow_best.pt)
  is the expected checkpoint path for the release artifact (~123 MB),
  trained on a mixed reasoning corpus and auto-discovered when
  `--method tflow` is used.

## Results

Our model achieves the following performance on:

### Reasoning Evaluation Across Five Public Benchmarks

Accuracy (%) on the full test split of each benchmark, with the
configuration shipped in this release (Qwen3-4B backbone,
`do_sample=True`, `temperature=0.6`, `top_p=0.95`, seed 42); pass@1
under the EvalPlus extended test harness for HumanEval+ / MBPP+.

| Model name | GSM8K | MATH  | MMLU  | HumanEval+ | MBPP+ |
| ---------- | :---: | :---: | :---: | :--------: | :---: |
| baseline   | 84.99 | 16.18 | 58.99 | 56.71      | 59.79 |
| textmas    | 93.78 | 26.47 | 71.50 | 75.00      | 68.52 |
| **tflow**  | 92.12 | 23.16 | 66.97 | 65.24      | 67.20 |

## Contributing

Issues and pull requests are welcome for reproducibility problems,
documentation fixes, and evaluation bugs. The code is released under
the Apache License 2.0; see [`LICENSE`](LICENSE) for the license text.

## Citation

```bibtex
@misc{tflow2026,
  title  = {Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update Your Weights},
  author = {TODO},
  year   = {2026},
  note   = {BibTeX will be updated after the public arXiv record is available}
}
```
