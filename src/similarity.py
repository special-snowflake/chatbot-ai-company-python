"""Vector helpers.

Python port of ``src/similarity.js``.
"""

from __future__ import annotations

import math
from typing import Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return the cosine similarity of two equal-length vectors.

    Mirrors ``cosineSimilarity`` in similarity.js exactly: mismatched or empty
    inputs yield ``0`` and a zero-magnitude vector yields ``0`` rather than a
    ``ZeroDivisionError``.
    """
    if len(left) != len(right) or len(left) == 0:
        return 0.0

    dot = 0.0
    left_magnitude = 0.0
    right_magnitude = 0.0
    for index in range(len(left)):
        dot += left[index] * right[index]
        left_magnitude += left[index] ** 2
        right_magnitude += right[index] ** 2

    denominator = math.sqrt(left_magnitude) * math.sqrt(right_magnitude)
    return 0.0 if denominator == 0 else dot / denominator
