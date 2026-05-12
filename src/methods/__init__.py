from src.methods.base import BaseMethod
from src.methods.baseline import BaselineMethod
from src.methods.textmas import TextMASMethod
from src.methods.tflow import TFlowMethod


METHOD_REGISTRY = {
    "baseline": BaselineMethod,
    "textmas": TextMASMethod,
    "tflow": TFlowMethod,
}


def build_method(config: dict) -> BaseMethod:
    """Factory function to instantiate a method from config."""
    method_name = config["method"]
    if method_name not in METHOD_REGISTRY:
        raise ValueError(
            f"Unknown method: {method_name}. "
            f"Available: {list(METHOD_REGISTRY.keys())}"
        )
    return METHOD_REGISTRY[method_name](config)
