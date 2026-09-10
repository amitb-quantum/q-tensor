"""Small, independently tested statistical comparison utilities."""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable, Iterable, Mapping


def probabilities(samples: Iterable[Hashable] | Mapping[Hashable, int]) -> dict[Hashable, float]:
    counts = Counter(samples) if not isinstance(samples, Mapping) else Counter(samples)
    total = sum(counts.values())
    if total <= 0:
        raise ValueError("at least one sample is required")
    if any(count < 0 for count in counts.values()):
        raise ValueError("sample counts must be non-negative")
    return {key: count / total for key, count in counts.items() if count}


def total_variation_distance(
    left: Iterable[Hashable] | Mapping[Hashable, int],
    right: Iterable[Hashable] | Mapping[Hashable, int],
) -> float:
    p = probabilities(left)
    q = probabilities(right)
    support = p.keys() | q.keys()
    return 0.5 * sum(abs(p.get(key, 0.0) - q.get(key, 0.0)) for key in support)
