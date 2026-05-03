"""Recipe -> fixed-length feature vector for ML training and inference.

Feature layout (deterministic, derived from the shortlist order):
    [g_<key1>, g_<key2>, ..., g_<keyN>, age_years]

We deliberately use raw ingredient grams + age and let the model learn
the nutrient transformation itself. That makes the RF-vs-GBR comparison
in D4 informative (both models have to learn the same non-trivial
mapping). If we handed them the per-100g-DM nutrient vector directly,
the score would be a near-trivial function of the input and the two
methods would tie at ceiling.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.data.ingredient_shortlist import SHORTLIST

INGREDIENT_KEYS: list[str] = [s.key for s in SHORTLIST]
KEY_INDEX: dict[str, int] = {k: i for i, k in enumerate(INGREDIENT_KEYS)}
FEATURE_NAMES: list[str] = [f"g_{k}" for k in INGREDIENT_KEYS] + ["age_years"]
N_FEATURES: int = len(FEATURE_NAMES)


def recipe_to_vector(recipe: dict[str, float], age_years: float) -> np.ndarray:
    """Encode a single recipe as a length-N_FEATURES float vector."""
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for key, grams in recipe.items():
        idx = KEY_INDEX.get(key)
        if idx is None:
            raise KeyError(f"unknown ingredient key: {key}")
        vec[idx] = float(grams)
    vec[-1] = float(age_years)
    return vec


def recipes_to_matrix(
    recipes: Sequence[dict[str, float]], ages: Iterable[float]
) -> np.ndarray:
    """Stack many encoded recipes into a 2-D (n_samples, N_FEATURES) matrix."""
    ages_list = list(ages)
    if len(ages_list) != len(recipes):
        raise ValueError("recipes and ages must have the same length")
    out = np.zeros((len(recipes), N_FEATURES), dtype=np.float32)
    for i, (r, a) in enumerate(zip(recipes, ages_list)):
        out[i] = recipe_to_vector(r, a)
    return out
