"""Dirichlet-Categorical label fusion.

Each tracked object owns an alpha vector over the Layer-0 vocabulary.
Updates are conjugate: alpha_new = alpha_old + weight * score_vector.
"""
from __future__ import annotations

import numpy as np


def init_alpha(vocab_size: int, prior: float = 0.5) -> np.ndarray:
    """Symmetric Dirichlet prior. 0.5 = Jeffreys prior (mildly informative)."""
    return np.full(vocab_size, prior, dtype=np.float32)


def update_alpha(
    alpha: np.ndarray,
    score_vector: np.ndarray,
    weight: float = 1.0,
) -> np.ndarray:
    """Conjugate Dirichlet-Categorical update.

    `weight` is typically the detector confidence in [0, 1]; weak detections
    move the posterior less than strong ones.
    """
    return alpha + float(weight) * score_vector.astype(np.float32)


def predict(alpha: np.ndarray) -> tuple[int, float]:
    """Return (argmax index, normalized Shannon entropy) of the posterior mean."""
    p = alpha / alpha.sum()
    eps = 1e-12
    h = -(p * np.log(p + eps)).sum() / np.log(len(p))
    return int(np.argmax(p)), float(h)


def topk(alpha: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    """Top-k (index, posterior-mean probability) pairs. For debugging."""
    p = alpha / alpha.sum()
    idx = np.argsort(p)[::-1][:k]
    return [(int(i), float(p[i])) for i in idx]
