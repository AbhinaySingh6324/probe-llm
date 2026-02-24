from probe.properties.consistency import Consistency
from probe.properties.invariance import Invariance
from probe.properties.negation import NegationCoherence
from probe.properties.robustness import Robustness
from probe.core.properties import Property

PROPERTY_REGISTRY: dict[str, type[Property]] = {
    "consistency": Consistency,
    "invariance": Invariance,
    "negation": NegationCoherence,
    "robustness": Robustness,
}


def get_property(name: str, **kwargs) -> Property:
    cls = PROPERTY_REGISTRY.get(name.lower())
    if cls is None:
        available = ", ".join(PROPERTY_REGISTRY.keys())
        raise ValueError(f"Unknown property: {name}. Available: {available}")
    return cls(**kwargs)
