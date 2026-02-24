"""
probe — Behavioral testing framework for LLMs.

Usage:
    from probe import quick_test, print_summary
    results = quick_test(
        model="openai:gpt-4o-mini",
        inputs=["What is the capital of France?"],
        properties=["consistency"],
    )
    print_summary(results)
"""

__version__ = "0.1.0"

from probe.core.models import ProbeResult, SuiteResult, Verdict
from probe.core.runner import run_suite_sync
from probe.core.reporter import print_summary, export_json
from probe.providers import get_provider
from probe.properties import get_property, PROPERTY_REGISTRY


def quick_test(
    model: str,
    inputs: list[str],
    properties: list[str] | None = None,
    threshold: float = 0.8,
    concurrency: int = 5,
) -> SuiteResult:
    """One-liner to run behavioral tests."""
    if properties is None:
        properties = ["consistency"]
    provider = get_provider(model)
    props = [get_property(name, threshold=threshold) for name in properties]
    return run_suite_sync(provider, inputs, props, concurrency=concurrency)


def compare_models(
    model_a: str,
    model_b: str,
    inputs: list[str],
    properties: list[str] | None = None,
    threshold: float = 0.8,
) -> tuple[SuiteResult, SuiteResult]:
    """Compare two models on the same behavioral tests."""
    if properties is None:
        properties = ["consistency"]
    prov_a, prov_b = get_provider(model_a), get_provider(model_b)
    props_a = [get_property(name, threshold=threshold) for name in properties]
    props_b = [get_property(name, threshold=threshold) for name in properties]
    sa = run_suite_sync(prov_a, inputs, props_a)
    sb = run_suite_sync(prov_b, inputs, props_b)
    return sa, sb
