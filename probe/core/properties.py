from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probe.providers.base import LLMProvider

from probe.core.models import ProbeResult


@dataclass
class PropertyConfig:
    threshold: float = 0.8
    comparator: str = "embedding"


class Property(ABC):
    name: str = "base_property"

    def __init__(self, config: PropertyConfig | None = None):
        self.config = config or PropertyConfig()

    @abstractmethod
    async def test(self, input_text: str, provider: "LLMProvider") -> ProbeResult:
        ...

    def _get_comparator(self):
        from probe.core.comparators import EmbeddingSimilarity, ExactMatch, ContainsMatch
        mapping = {
            "embedding": EmbeddingSimilarity,
            "exact": ExactMatch,
            "contains": ContainsMatch,
        }
        return mapping.get(self.config.comparator, EmbeddingSimilarity)()
