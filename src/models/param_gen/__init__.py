"""ParameterGenerator module used by the TFlow adapter.

The ``ParameterGenerator`` maps a sender agent's aggregated hidden states
into a LoRA factor pair ``(A, B)`` for every targeted ``nn.Linear`` of the
receiver backbone; the factor pairs are then convex-combined across senders
and applied through forward hooks at inference time.
"""

from .config import ParameterGeneratorConfig
from .generator import ParameterGenerator

# Register the config so ``transformers.AutoConfig`` resolves it when
# loading from a Hugging Face repo or a local folder.
try:
    from transformers import AutoConfig

    AutoConfig.register(ParameterGeneratorConfig.model_type, ParameterGeneratorConfig)
except Exception:
    pass

__all__ = ["ParameterGenerator", "ParameterGeneratorConfig"]
