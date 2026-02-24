from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"


@dataclass
class ProbeResult:
    input: str
    property_name: str
    verdict: Verdict
    score: float
    details: dict[str, Any] = field(default_factory=dict)
    original_output: str = ""
    variant_outputs: list[str] = field(default_factory=list)
    elapsed_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS


@dataclass
class SuiteResult:
    results: list[ProbeResult] = field(default_factory=list)
    model_name: str = ""
    total_elapsed_ms: float = 0.0

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.FAIL)

    @property
    def errors(self) -> int:
        return sum(1 for r in self.results if r.verdict == Verdict.ERROR)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def failures(self) -> list[ProbeResult]:
        return [r for r in self.results if not r.passed]

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "errors": self.errors,
            "pass_rate": round(self.pass_rate, 4),
            "elapsed_ms": round(self.total_elapsed_ms, 1),
            "results": [
                {
                    "input": r.input[:80],
                    "property": r.property_name,
                    "verdict": r.verdict.value,
                    "score": round(r.score, 4),
                    "details": r.details,
                }
                for r in self.results
            ],
        }
