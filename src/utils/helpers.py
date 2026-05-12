"""Answer extraction, normalisation, and per-dataset correctness checks.

Minerva Math scoring follows ``EleutherAI/lm-evaluation-harness``
(``lm_eval/tasks/minerva_math/utils.py``), which itself reproduces
Appendix D of Lewkowycz et al. (2022).  Code-task scoring runs the
extracted ``python`` block against the gold test harness in a subprocess
with a 10-second timeout.
"""

import math
import os
import re
import random
import logging
import traceback
from multiprocessing import Process, Manager
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional heavy dependencies for Minerva Math equivalence checking
# ---------------------------------------------------------------------------
try:
    import sympy
    from sympy.parsing.latex import parse_latex

    _HAS_SYMPY = True
except ImportError:
    _HAS_SYMPY = False

try:
    from math_verify import parse as mv_parse, verify as mv_verify

    _HAS_MATH_VERIFY = True
except ImportError:
    _HAS_MATH_VERIFY = False


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_backbone_tag(config: dict) -> str:
    """Sanitize HuggingFace model id for checkpoint filenames (e.g. Qwen/Qwen3-8B → Qwen_Qwen3-8B)."""
    name = str(config.get("model", {}).get("name", "unknown"))
    tag = re.sub(r"[^\w\-]", "_", name) or "unknown"
    return tag


def extract_answer(text: str, pattern: str = r"####\s*([\-\d\.\,]+)") -> Optional[str]:
    """Extract final answer from model output using regex."""
    matches = re.findall(pattern, text)
    if matches:
        answer = matches[-1].replace(",", "").strip()
        return answer
    return None


# ======================================================================
# Minerva Math utilities (aligned with lm-evaluation-harness)
# Source: lm_eval/tasks/minerva_math/utils.py
# ======================================================================

def last_boxed_only_string(string: str) -> Optional[str]:
    r"""Return the last ``\boxed{...}`` or ``\fbox{...}`` substring, correctly
    handling nested braces.  Taken verbatim from lm-evaluation-harness."""
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        return None
    return string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    r"""Strip the ``\boxed{...}`` wrapper, returning the inner content."""
    if not s:
        return ""
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] == left:
            return s[len(left) :]
    left = "\\boxed{"
    if s[: len(left)] == left and s[-1] == "}":
        return s[len(left) : -1]
    return s


_MINERVA_SUBSTITUTIONS: List[Tuple[str, str]] = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]

_MINERVA_REMOVED_EXPRESSIONS: List[str] = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft",
    "hours", "km", "units", "\\ldots", "sue", "points", "feet",
    "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds",
    "meters", "meals", "edges", "students", "childrentickets",
    "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}",
    '"', "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    """Normalize a final answer to a quantitative reasoning question.

    Copied character for character from Appendix D of Lewkowycz et al. (2022),
    via ``lm_eval/tasks/minerva_math/utils.py``.
    """
    final_answer = final_answer.split("=")[-1]

    for before, after in _MINERVA_SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in _MINERVA_REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")

    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)

    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")

    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")

    return final_answer


def get_unnormalized_answer(text: str) -> str:
    """Extract answer from the Minerva "Final Answer: The final answer is …"
    pattern used in the 4-shot exemplars."""
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.",
        text,
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


_LATEX_SCI_RE = re.compile(
    r"^([+-]?\d+(?:\.\d+)?)"
    r"\s*(?:\\times|\\cdot)\s*"
    r"10\s*\^\s*\{?\s*([+-]?\d+)\s*\}?"
    r"$"
)


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a float, handling both Python scientific
    notation (``4.5e33``) and LaTeX (``4.5\\times10^{33}``).

    Also strips common LaTeX wrappers like ``\\(...\\)`` and ``$...$``.
    """
    s = s.strip()
    for wrap_l, wrap_r in [("\\(", "\\)"), ("$", "$")]:
        if s.startswith(wrap_l) and s.endswith(wrap_r):
            s = s[len(wrap_l):-len(wrap_r)].strip()
    try:
        return float(s)
    except (ValueError, OverflowError):
        pass
    m = _LATEX_SCI_RE.match(s)
    if m:
        try:
            return float(f"{m.group(1)}e{m.group(2)}")
        except (ValueError, OverflowError):
            pass
    s_clean = s.replace("{", "").replace("}", "")
    try:
        return float(s_clean)
    except (ValueError, OverflowError):
        pass
    return None


def is_equiv(x1: str, x2: str) -> bool:
    """Check whether two *normalized* LaTeX strings are mathematically
    equivalent via ``sympy.parse_latex`` + ``simplify``, with a numeric
    fallback for scientific notation and plain numbers.

    Falls back to exact string comparison when sympy / antlr4 are unavailable.
    """
    if x1 == x2:
        return True

    n1 = _try_parse_number(x1)
    n2 = _try_parse_number(x2)
    if n1 is not None and n2 is not None:
        if n1 == n2:
            return True
        if n1 != 0 and n2 != 0 and math.isclose(n1, n2, rel_tol=1e-4):
            return True

    if not _HAS_SYMPY:
        return False
    try:
        parsed_x1 = parse_latex(x1)
        parsed_x2 = parse_latex(x2)
    except Exception:
        logger.debug("couldn't parse one of %s or %s", x1, x2)
        return False
    try:
        diff = parsed_x1 - parsed_x2
    except TypeError:
        logger.debug("couldn't subtract %s and %s", x1, x2)
        return False
    try:
        if sympy.simplify(diff) == 0:
            return True
        return False
    except Exception:
        logger.debug("trouble simplifying when comparing %s and %s", x1, x2)
        return False


def minerva_math_is_correct(pred_text: str, gold_answer: str) -> bool:
    """Score a single minerva_math sample aligned with lm-evaluation-harness.

    ``gold_answer`` should already be the *normalized* gold string produced by
    ``normalize_final_answer(remove_boxed(last_boxed_only_string(...)))``.

    Extraction pipeline (first match wins):
    1. ``get_unnormalized_answer`` — "Final Answer: The final answer is ... I hope it is correct."
    2. ``last_boxed_only_string`` + ``remove_boxed`` — extract from ``\\boxed{...}``
    3. ``math_verify`` fallback (parses full output).

    Each extracted answer is compared with ``gold_answer`` via ``is_equiv`` (sympy).
    """
    # Truncate at "Problem:" to mirror lm_eval's generate_until stop sequence
    stop_idx = pred_text.find("Problem:")
    if stop_idx != -1:
        pred_text = pred_text[:stop_idx]

    # --- 1. "Final Answer" pattern (lm_eval exact_match primary) ---
    unnorm = get_unnormalized_answer(pred_text)
    if unnorm != "[invalidanswer]":
        pred_normalized = normalize_final_answer(unnorm)
        if is_equiv(pred_normalized, gold_answer):
            return True

    # --- 2. \boxed{} extraction (most common for chat models) ---
    boxed = last_boxed_only_string(pred_text)
    if boxed is not None:
        pred_from_box = normalize_final_answer(remove_boxed(boxed))
        if is_equiv(pred_from_box, gold_answer):
            return True

    # --- 3. math_verify (parses full output, handles diverse formats) ---
    if _HAS_MATH_VERIFY:
        try:
            gold_boxed = "\\boxed{" + gold_answer + "}"
            result = mv_verify(
                gold=mv_parse(gold_boxed),
                target=mv_parse(pred_text),
            )
            if result:
                return True
        except Exception:
            pass

    return False


def extract_minerva_math_prediction(pred_text: str) -> str:
    """Extract and normalize the predicted answer for minerva_math.

    Tries "Final Answer" pattern first, then \\boxed{} extraction.
    """
    stop_idx = pred_text.find("Problem:")
    if stop_idx != -1:
        pred_text = pred_text[:stop_idx]

    # 1. "Final Answer" pattern
    unnorm = get_unnormalized_answer(pred_text)
    if unnorm != "[invalidanswer]":
        return normalize_final_answer(unnorm)

    # 2. \boxed{} extraction
    boxed = last_boxed_only_string(pred_text)
    if boxed is not None:
        return normalize_final_answer(remove_boxed(boxed))

    return "[invalidanswer]"


def resolve_system_prompt_for_dataset(method_config: dict, dataset_str: str) -> str:
    """Pick system prompt for eval/train based on `data.dataset` (supports comma-separated mix).

    Uses `method_config.system_prompt_by_dataset` with keys matching `datasets.resolve_dataset_name`;
    falls back to `method_config.system_prompt`.
    """
    default = method_config.get(
        "system_prompt",
        "You are a helpful assistant. Solve step by step. Put your final answer after ####.",
    )
    by_ds = method_config.get("system_prompt_by_dataset")
    if not by_ds or not dataset_str:
        return default
    # Lazy import: datasets imports helpers for extract_*
    from src.data.datasets import resolve_dataset_name

    first = dataset_str.split(",")[0].strip()
    if not first:
        return default
    key = resolve_dataset_name(first)
    return by_ds.get(key, default)


def normalize_answer(answer: Optional[str]) -> str:
    """Normalize numeric answer for comparison."""
    if answer is None or answer == "":
        return ""
    try:
        val = float(str(answer).replace(",", ""))
        if not math.isfinite(val):
            return str(answer).strip().lower()
        if val == int(val):
            return str(int(val))
        return str(val)
    except (ValueError, TypeError, OverflowError):
        return str(answer).strip().lower()


def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract answer from ``\\boxed{}`` (last box) or fall back to the last number in the text."""
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1]
    return None


# Sources that share MCQ-style answer extraction (letter A–D / a–d).
LATENTMAS_MCQ_SOURCES = frozenset({"arc_easy", "arc_challenge", "gpqa", "medqa", "mmlu"})

# Code benchmarks: extract ```python``` then run with the gold harness in a subprocess.
LATENTMAS_CODE_TASK_SOURCES = frozenset({"mbppplus", "humanevalplus"})


MINERVA_MATH_SOURCES = frozenset({"minerva_math"})


def extract_latentmas_eval_prediction(
    pred_text: str,
    source: str,
    *,
    extract_answer_regex: str = r"####\s*([\-\d\.\,]+)",
    extract_answer_regex_by_dataset: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Unified prediction string for eval, dispatched by ``source``.

    - Minerva Math: lm-evaluation-harness extraction (``get_unnormalized_answer`` → ``normalize_final_answer``).
    - Code tasks: last ```python``` block (or None).
    - AIME: ``extract_gsm8k_answer`` (``\\boxed`` / last number).
    - MCQ: ``extract_latentmas_mcq_prediction``.
    - gsm8k / math / etc.: ``extract_gsm8k_answer``, then ``extract_answer`` with
      per-dataset regex fallback (default GSM8K-style ``####`` number).
    """
    by = extract_answer_regex_by_dataset or {}
    regex_fb = by.get(source, extract_answer_regex)

    if source in MINERVA_MATH_SOURCES:
        return extract_minerva_math_prediction(pred_text)

    if source in LATENTMAS_CODE_TASK_SOURCES:
        return extract_markdown_python_block(pred_text)

    if source in ("aime2024", "aime2025"):
        return extract_gsm8k_answer(pred_text)

    if source in LATENTMAS_MCQ_SOURCES:
        return extract_latentmas_mcq_prediction(pred_text)

    pred = extract_gsm8k_answer(pred_text)
    if pred is None:
        pred = extract_answer(pred_text, regex_fb)
    return pred


def latentmas_eval_is_correct(
    pred_text: str,
    gold_answer: str,
    source: str,
    *,
    extract_answer_regex: str = r"####\s*([\-\d\.\,]+)",
    extract_answer_regex_by_dataset: Optional[Dict[str, str]] = None,
) -> bool:
    """Whether ``pred_text`` matches ``gold_answer`` per task-specific rules.

    For ``minerva_math``, scoring is fully aligned with lm-evaluation-harness
    (sympy equivalence + optional math_verify).
    """
    if source in MINERVA_MATH_SOURCES:
        return minerva_math_is_correct(pred_text, gold_answer)

    if source in LATENTMAS_CODE_TASK_SOURCES:
        pred_code = extract_markdown_python_block(pred_text)
        if pred_code is None:
            return False
        code_to_run = pred_code + "\n" + gold_answer
        ok, _ = run_with_timeout(code_to_run, timeout=10)
        return ok

    if source in ("aime2024", "aime2025"):
        pred = extract_gsm8k_answer(pred_text)
        pred_norm = normalize_answer(pred) if pred else ""
        gold_norm = str(gold_answer).strip()
        try:
            return int(pred_norm) == int(gold_norm)
        except ValueError:
            return False

    if source in LATENTMAS_MCQ_SOURCES:
        pred = extract_latentmas_mcq_prediction(pred_text)
        pred_norm = normalize_answer(pred) if pred else ""
        gold_norm = normalize_answer(gold_answer)
        return bool(pred_norm and gold_norm and pred_norm == gold_norm)

    pred = extract_latentmas_eval_prediction(
        pred_text,
        source,
        extract_answer_regex=extract_answer_regex,
        extract_answer_regex_by_dataset=extract_answer_regex_by_dataset,
    )
    pred_norm = normalize_answer(pred) if pred else ""
    gold_norm = normalize_answer(gold_answer)
    return bool(pred_norm and gold_norm and pred_norm == gold_norm)


def extract_latentmas_mcq_prediction(text: str) -> Optional[str]:
    """MCQ-style extraction for ARC / GPQA / MedQA / MMLU.

    When ``\\boxed{...}`` is present: use the last box; if the box contains a
    number take it, otherwise the stripped box content (e.g. ``A``).

    Without ``\\boxed{}``, falling back to the last number in the text would be
    inappropriate for letter MCQ; we instead try (in order): last standalone
    letter ``A``–``D``, then ``####`` followed by a letter.
    """
    boxes = re.findall(r"\\boxed\{([^}]*)\}", text)
    if boxes:
        content = boxes[-1]
        number = re.search(r"[-+]?\d+(?:\.\d+)?", content)
        return number.group(0) if number else content.strip()
    letters = re.findall(r"\b([A-Da-d])\b", text)
    if letters:
        return letters[-1]
    m = re.search(r"####\s*([A-Da-d])\b", text, re.IGNORECASE)
    if m:
        return m.group(1)
    return None


def extract_markdown_python_block(text: str) -> Optional[str]:
    """Extract the last ```python ... ``` block from ``text``."""
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[-1].strip()
    return None


def run_with_timeout(code: str, timeout: int = 10) -> Tuple[bool, Optional[str]]:
    """Execute ``code`` in a subprocess with a wall-clock timeout; return ``(ok, error_msg)``."""
    def worker(ns: dict, code_str: str) -> None:
        try:
            local_ns: dict = {}
            exec(code_str, local_ns)
            ns["ok"] = True
            ns["error"] = None
        except Exception:
            ns["ok"] = False
            ns["error"] = traceback.format_exc()

    with Manager() as manager:
        ns = manager.dict()
        p = Process(target=worker, args=(ns, code))
        p.start()
        p.join(timeout=timeout)
        if p.is_alive():
            p.terminate()
            ns["ok"] = False
            ns["error"] = f"TimeoutError: Execution exceeded {timeout} seconds"
        return bool(ns.get("ok", False)), ns.get("error")


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def setup_logging(output_dir: str, level: int = logging.INFO) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(output_dir, "run.log")),
        ],
    )
    for noisy in ("httpx", "httpcore", "urllib3", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)