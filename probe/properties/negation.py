import time
from probe.core.properties import Property, PropertyConfig
from probe.core.models import ProbeResult, Verdict
from probe.core.transforms import NegationTransform
from probe.providers.base import LLMProvider


class NegationCoherence(Property):
    """Negate the question → answer should logically flip (low similarity = pass)."""

    name = "negation_coherence"

    def __init__(self, threshold: float = 0.7, comparator: str = "embedding"):
        super().__init__(PropertyConfig(threshold=threshold, comparator=comparator))
        self.transform = NegationTransform()

    async def test(self, input_text: str, provider: LLMProvider) -> ProbeResult:
        t0 = time.perf_counter()
        comp = self._get_comparator()

        original = await provider.generate(input_text)
        negated_inputs = await self.transform.apply(input_text, provider)
        if not negated_inputs:
            return ProbeResult(input=input_text, property_name=self.name,
                               verdict=Verdict.ERROR, score=0.0,
                               details={"error": "Negation transform failed"})

        negated_output = await provider.generate(negated_inputs[0])
        sim = comp.similarity(original, negated_output)
        divergence = 1.0 - sim
        passed = divergence >= self.config.threshold

        return ProbeResult(
            input=input_text, property_name=self.name,
            verdict=Verdict.PASS if passed else Verdict.FAIL,
            score=round(divergence, 4), original_output=original,
            variant_outputs=[negated_output],
            elapsed_ms=(time.perf_counter() - t0) * 1000,
            details={"threshold": self.config.threshold,
                     "similarity": round(sim, 4),
                     "divergence": round(divergence, 4),
                     "negated_input": negated_inputs[0]},
        )
