from abc import ABC, abstractmethod


class LLMProvider(ABC):
    model_name: str = "unknown"

    @abstractmethod
    async def generate(self, prompt: str, temperature: float = 0.0) -> str:
        ...

    @abstractmethod
    async def generate_batch(self, prompts: list[str], temperature: float = 0.0) -> list[str]:
        ...
