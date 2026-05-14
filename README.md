# TFlow: Weight-Space Communication for Multi-Agent LLMs

This repository contains the official code release for
**[Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update
Your Weights](https://arxiv.org/pdf/2605.13839)**.

TFlow (Thought Flow) is a weight-space communication framework for
multi-agent LLMs. Instead of asking sender agents to write natural
language messages into the receiver's context, TFlow compiles sender
hidden states into transient, receiver-specific LoRA perturbations.
These perturbations are fused and applied only during the receiver's
generation, enabling instance-level adaptation without permanently
changing the model weights or increasing the receiver's text context.

With three frozen Qwen3-4B agents, TFlow improves over a standalone
receiver by up to 8.5 accuracy points while reducing processed tokens by
up to 32.69%. Compared with a text-based three-agent baseline, it reduces
processed tokens by up to 83.27% and wall-clock inference time by up to
4.6x, while maintaining competitive accuracy on four of five benchmarks.

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
python main.py --method tflow --dataset gsm8k
python main.py --method tflow --dataset minerva_math 
python main.py --method tflow --dataset mmlu       
python main.py --method tflow --dataset humaneval   
python main.py --method tflow --dataset mbpp      
```

You can also run all five datasets through the convenience script:

```eval
MAX_SAMPLES=-1 GENERATE_BS=20 bash scripts/run_eval.sh tflow
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

Table 1 in the paper reports accuracy (Acc., %), average total processed
tokens (Token), and end-to-end wall-clock inference time (Speed, seconds)
across five benchmarks. `Single` is the standalone receiver, `TextMAS`
is the text-channel multi-agent baseline, and `TFlow` is the proposed
weight-space communication method.

| Task | Metric | Single | TextMAS | TFlow |
| --- | --- | ---: | ---: | ---: |
| MMLU-Redux | Acc. | 58.99 | 71.50 | 66.97 |
| MMLU-Redux | Token | 1079 | 4825 | 998 |
| MMLU-Redux | Speed | 8226 | 36450 | 9784 |
| GSM8K | Acc. | 84.99 | 93.78 | 92.12 |
| GSM8K | Token | 1337 | 5381 | 900 |
| GSM8K | Speed | 6230 | 27256 | 5953 |
| MATH | Acc. | 16.18 | 26.47 | 23.16 |
| MATH | Token | 2782 | 8188 | 2242 |
| MATH | Speed | 1984 | 5213 | 2258 |
| MBPP+ | Acc. | 59.79 | 68.52 | 67.20 |
| MBPP+ | Token | 1533 | 5500 | 1301 |
| MBPP+ | Speed | 1998 | 5796 | 2395 |
| HumanEval+ | Acc. | 56.71 | 75.00 | 65.24 |
| HumanEval+ | Token | 1756 | 5879 | 1662 |
| HumanEval+ | Speed | 872 | 2512 | 1065 |

Overall, TFlow consistently outperforms the single-agent baseline while
using fewer processed tokens. Against TextMAS, it cuts token consumption
by 71-83% and provides 2.3-4.6x wall-clock speed-ups, with the largest
accuracy gap appearing on HumanEval+.

## Contributing

Issues and pull requests are welcome for reproducibility problems,
documentation fixes, and evaluation bugs. The code is released under
the Apache License 2.0; see [`LICENSE`](LICENSE) for the license text.

## Citation

```bibtex
@misc{tflow2026good,
  title         = {Good Agentic Friends Do Not Just Give Verbal Advice: They Can Update Your Weights},
  author        = {TODO},
  year          = {2026},
  eprint        = {2605.13839},
  archivePrefix = {arXiv},
  primaryClass  = {cs.AI},
  url           = {https://arxiv.org/pdf/2605.13839}
}
```
