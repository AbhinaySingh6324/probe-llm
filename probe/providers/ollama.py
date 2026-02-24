from __future__ import annotations
import httpx
from probe.providers.base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model_name = model
        self._base_url = base_url

    async def generate(self, prompt: str, temperature: float = 0.0) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
            )
            resp.raise_for_status()
            return resp.json().get("response", "")

    async def generate_batch(self, prompts: list[str], temperature: float = 0.0) -> list[str]:
        results = []
        for p in prompts:
            results.append(await self.generate(p, temperature))
        return results
