import time
from probe.core.properties import Property, PropertyConfig
from probe.core.models import ProbeResult, Verdict
from probe.core.transforms import TypoTransform
from probe.providers.base import LLMProvider


class Robustness(Property):
    """Inject typos → output should remain semantically equivalent."""

    name = "robustness"

    def __init__(self, n_typos: int = 3, threshold: float = 0.85, comparator: str = "embedding"):
        super().__init__(PropertyConfig(threshold=threshold, comparator=comparator))
        self.transform = TypoTransform(n=n_typos)

    async def test(self, input_text: str, provider: LLMProvider) -> ProbeResult:
        t0 = time.perf_counter()
        comp = self._get_comparator()

        original = await provider.generate(input_text)
        typo_variants = await self.transform.apply(input_text)
        variant_outputs = await provider.generate_batch(typo_variants)

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
                     "typo_inputs": typo_variants},
        )
