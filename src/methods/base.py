"""Abstract base class for the inference methods."""

from __future__ import annotations

import abc
import logging
from typing import Any

logger = logging.getLogger(__name__)


class BaseMethod(abc.ABC):
    """Abstract base for the three inference methods (baseline / textmas / tflow)."""

    def __init__(self, config: dict):
        self.config = config
        self.method_config = config.get("method_config", {})
        self.model_config = config.get("model", {})

        dataset = config.get("data", {}).get("dataset", "")
        by_ds = self.model_config.get("max_new_tokens_by_dataset", {})
        if dataset in by_ds:
            self.model_config["max_new_tokens"] = by_ds[dataset]
            logger.info(
                "max_new_tokens overridden to %d for dataset '%s'",
                by_ds[dataset], dataset,
            )

    @abc.abstractmethod
    def setup(self) -> None:
        """Initialise agents, ParameterGenerator, and any inference-time state."""
        ...

    @abc.abstractmethod
    def solve(self, question: str) -> dict[str, Any]:
        """Answer a single question.

        Returns a dict with at least:

        - ``answer``: final response text;
        - ``reasoning``: rendered reasoning trace (may be identical to
          ``answer``);
        - ``metadata``: timing, token usage, and a
          ``truncated_by_max_new_tokens`` flag.
        """
        ...

    @abc.abstractmethod
    def solve_batch(self, questions: list[str]) -> list[dict[str, Any]]:
        """Solve a batch of questions."""
        ...