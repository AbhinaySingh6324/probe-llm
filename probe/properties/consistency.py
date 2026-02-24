import time
from probe.core.properties import Property, PropertyConfig
from probe.core.models import ProbeResult, Verdict
from probe.core.transforms import ParaphraseTransform
from probe.providers.base import LLMProvider


class Consistency(Property):
    """Rephrase input N ways → check all outputs are semantically equivalent."""

    name = "consistency"

    def __init__(self, n_rephrasings: int = 5, threshold: float = 0.8, comparator: str = "embedding"):
        super().__init__(PropertyConfig(threshold=threshold, comparator=comparator))
        self.transform = ParaphraseTransform(n=n_rephrasings)

    async def test(self, input_text: str, provider: LLMProvider) -> ProbeResult:
        t0 = time.perf_counter()
        comp = self._get_comparator()

        original = await provider.generate(input_text)
        variants = await self.transform.apply(input_text, provider)
        if not variants:
            return ProbeResult(input=input_text, property_name=self.name,
                               verdict=Verdict.ERROR, score=0.0,
                               details={"error": "No rephrasings generated"})

        variant_outputs = await provider.generate_batch(variants)
        scores = (comp.batch_similarity(original, variant_outputs)
                  if hasattr(comp, "batch_similarity")
                  else [comp.similarity(original, vo) for vo in variant_outputs])

        avg = sum(scores) / len(scores) if scores else 0.0
        pass_frac = sum(1 for s in scores if s >= self.config.threshold) / len(scores)
        passed = pass_frac >= self.config.threshold

        return ProbeResult(
            input=input_text, property_name=self.name,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=round(avg, 4), original_output=original,
            variant_outputs=variant_outputs,
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            details={"threshold": self.config.threshold,
                     "pairwise_scores": [round(s, 4) for s in scores],
                     "pass_fraction": round(pass_frac, 4),
                     "rephrasings": variants},
        )
