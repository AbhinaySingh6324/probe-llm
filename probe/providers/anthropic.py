from __future__ import annotations
import asyncio
import os
from probe.providers.base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-20250514", api_key: str | None = None):
        self.model_name = model
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic
            except ImportError:
                raise ImportError("Install anthropic: pip install probe-llm[anthropic]")
            self._client = AsyncAnthropic(api_key=self._api_key)
        return self._client

    async def generate(self, prompt: str, temperature: float = 0.0) -> str:
        client = self._get_client()
        resp = await client.messages.create(
            model=self.model_name,
            max_tokens=1024,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text if resp.content else ""

    async def generate_batch(self, prompts: list[str], temperature: float = 0.0) -> list[str]:
        return await asyncio.gather(*[self.generate(p, temperature) for p in prompts])
