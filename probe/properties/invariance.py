import time
from probe.core.properties import Property, PropertyConfig
from probe.core.models import ProbeResult, Verdict
from probe.core.transforms import EntitySwapTransform
from probe.providers.base import LLMProvider


class Invariance(Property):
    """Swap irrelevant entities → check output structure holds."""

    name = "invariance"

    def __init__(self, n_variants: int = 3, threshold: float = 0.8, comparator: str = "embedding"):
        super().__init__(PropertyConfig(threshold=threshold, comparator=comparator))
        self.transform = EntitySwapTransform(n=n_variants)

    async def test(self, input_text: str, provider: LLMProvider) -> ProbeResult:
        t0 = time.perf_counter()
        comp = self._get_comparator()

        original = await provider.generate(input_text)
        variants = await self.transform.apply(input_text, provider)
        if not variants:
            return ProbeResult(input=input_text, property_name=self.name,
                               verdict=Verdict.ERROR, score=0.0,
                               details={"error": "No entity-swap variants generated"})

        variant_outputs = await provider.generate_batch(variants)
        scores = (comp.batch_similarity(original, variant_outputs)
                  if hasattr(comp, "batch_similarity")
                  else [comp.similarity(original, vo) for vo in variant_outputs])

        avg = sum(scores) / len(scores) if scores else 0.0
        passed = avg >= self.config.threshold

        return ProbeResult(
            input=input_text, property_name=self.name,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=round(avg, 4), original_output=original,
            variant_outputs=variant_outputs,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            details={"threshold": self.config.threshold,
                     "pairwise_scores": [round(s, 4) for s in scores],
                     "entity_variants": variants},
        )
