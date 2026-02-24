from __future__ import annotations
import asyncio
import os
from probe.providers.base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        self.model_name = model
        self._api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
            except ImportError:
                raise ImportError("Install openai: pip install probe-llm[openai]")
            self._client = AsyncOpenAI(api_key=self._api_key)
        return self._client

    async def generate(self, prompt: str, temperature: float = 0.0) -> str:
        client = self._get_client()
        resp = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content or ""

    async def generate_batch(self, prompts: list[str], temperature: float = 0.0) -> list[str]:
        return await asyncio.gather(*[self.generate(p, temperature) for p in prompts])
