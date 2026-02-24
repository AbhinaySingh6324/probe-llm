from __future__ import annotations
import asyncio
import time as _time
from typing import TYPE_CHECKING

from probe.core.models import ProbeResult, SuiteResult, Verdict
from probe.core.properties import Property

if TYPE_CHECKING:
    from probe.providers.base import LLMProvider


async def _run_single(prop: Property, inp: str, provider: "LLMProvider") -> ProbeResult:
    try:
        return await prop.test(inp, provider)
    except Exception as e:
        return ProbeResult(
            input=inp, property_name=prop.name,
            verdict=Verdict.ERROR, score=0.0,
            details={"error": str(e)},
        )


async def run_suite(
    provider: "LLMProvider",
    inputs: list[str],
    properties: list[Property],
    concurrency: int = 5,
) -> SuiteResult:
    start = _time.perf_counter()
    sem = asyncio.Semaphore(concurrency)

    async def bounded(prop, inp):
        async with sem:
            return await _run_single(prop, inp, provider)

    tasks = [bounded(prop, inp) for inp in inputs for prop in properties]
    results = await asyncio.gather(*tasks)
    elapsed = (_time.perf_counter() - start) * 1000

    return SuiteResult(
        results=list(results),
        model_name=provider.model_name,
        total_elapsed_ms=elapsed,
    )


def run_suite_sync(
    provider: "LLMProvider",
    inputs: list[str],
    properties: list[Property],
    concurrency: int = 5,
) -> SuiteResult:
    return asyncio.run(run_suite(provider, inputs, properties, concurrency))
