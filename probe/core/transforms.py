from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from probe.providers.base import LLMProvider


class Transform(ABC):
    name: str = "base_transform"

    @abstractmethod
    async def apply(self, text: str, provider: "LLMProvider | None" = None) -> list[str]:
        ...


class ParaphraseTransform(Transform):
    name = "paraphrase"

    def __init__(self, n: int = 5):
        self.n = n

    async def apply(self, text: str, provider: "LLMProvider | None" = None) -> list[str]:
        if provider is None:
            raise ValueError("ParaphraseTransform requires an LLM provider")
        prompt = (
            f"Rephrase the following text in {self.n} different ways. "
            f"Each rephrasing must preserve the exact same meaning and intent. "
            f"Return ONLY the rephrasings, one per line, numbered 1-{self.n}.\n\n"
            f"Text: {text}"
        )
        raw = await provider.generate(prompt)
        lines = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = line.lstrip("0123456789").lstrip(".-)").strip()
            if cleaned:
                lines.append(cleaned)
        return lines[:self.n]


class EntitySwapTransform(Transform):
    name = "entity_swap"

    def __init__(self, n: int = 3):
        self.n = n

    async def apply(self, text: str, provider: "LLMProvider | None" = None) -> list[str]:
        if provider is None:
            raise ValueError("EntitySwapTransform requires an LLM provider")
        prompt = (
            f"Take this text and create {self.n} variants by swapping named entities "
            f"(person names, places, organizations, specific numbers) with different "
            f"but plausible alternatives. Keep the structure and intent identical.\n"
            f"Return ONLY the variants, one per line, numbered 1-{self.n}.\n\n"
            f"Text: {text}"
        )
        raw = await provider.generate(prompt)
        lines = []
        for line in raw.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            cleaned = line.lstrip("0123456789").lstrip(".-)").strip()
            if cleaned:
                lines.append(cleaned)
        return lines[:self.n]


class NegationTransform(Transform):
    name = "negation"

    async def apply(self, text: str, provider: "LLMProvider | None" = None) -> list[str]:
        if provider is None:
            raise ValueError("NegationTransform requires an LLM provider")
        prompt = (
            "Negate the following question or statement. If it asks 'Is X true?', "
            "change it to 'Is X NOT true?' or 'Is X false?'. Preserve the topic.\n"
            "Return ONLY the negated version, nothing else.\n\n"
            f"Text: {text}"
        )
        raw = await provider.generate(prompt)
        return [raw.strip()]


class TypoTransform(Transform):
    name = "typo"

    def __init__(self, n: int = 3):
        self.n = n

    async def apply(self, text: str, provider: "LLMProvider | None" = None) -> list[str]:
        import random
        variants = []
        for _ in range(self.n):
            chars = list(text)
            if len(chars) < 4:
                variants.append(text)
                continue
            idx = random.randint(1, len(chars) - 2)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            variants.append("".join(chars))
        return variants
