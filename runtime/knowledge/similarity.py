"""
QAIR Vector Similarity

Numerical utilities used by the semantic retrieval layer.
"""

from __future__ import annotations

import math


def cosine_similarity(
    left: list[float],
    right: list[float],
) -> float:
    """
    Calculate cosine similarity between two vectors.

    Returns a value between -1.0 and 1.0 for valid vectors.
    """

    if not left or not right:
        raise ValueError("Vectors cannot be empty.")

    if len(left) != len(right):
        raise ValueError(
            "Vectors must have the same dimensionality."
        )

    dot_product = sum(
        a * b
        for a, b in zip(left, right)
    )

    left_norm = math.sqrt(
        sum(value * value for value in left)
    )

    right_norm = math.sqrt(
        sum(value * value for value in right)
    )

    if left_norm == 0.0 or right_norm == 0.0:
        raise ValueError(
            "Vectors must have non-zero magnitude."
        )

    return dot_product / (left_norm * right_norm)