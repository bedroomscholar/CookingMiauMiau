"""Recipe -> fixed-length feature vector for ML training and inference.

Feature layout (deterministic, identity-free):
    [protein_g, fat_g, calcium_mg, phosphorus_mg, taurine_mg,
     magnesium_mg, sodium_mg, potassium_mg, ca_p_ratio, age_years]

All nutrient features are per-100g of dry matter — exactly what the rule
scorer's `per_100g_dm()` already computes from the recipe + the
ingredients DataFrame.

Why nutrient features instead of per-ingredient grams: the model now
generalises to *any* combination of ingredients (built-in or
user-added) as long as the merged ingredients table has nutrient rows
for them. Adding a new ingredient no longer invalidates the trained
model — the feature vector shape is fixed at 10 regardless of how many
ingredients exist.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from src.nutrition.scorer import per_100g_dm

NUTRIENT_FEATURES: list[str] = [
    "protein_g",
    "fat_g",
    "calcium_mg",
    "phosphorus_mg",
    "taurine_mg",
    "magnesium_mg",
    "sodium_mg",
    "potassium_mg",
    "ca_p_ratio",
]
FEATURE_NAMES: list[str] = NUTRIENT_FEATURES + ["age_years"]
N_FEATURES: int = len(FEATURE_NAMES)


def recipe_to_vector(
    recipe: dict[str, float], age_years: float, ingredients: pd.DataFrame
) -> np.ndarray:
    """Encode a single recipe as a length-N_FEATURES float vector."""
    values = per_100g_dm(recipe, ingredients)
    vec = np.zeros(N_FEATURES, dtype=np.float32)
    for i, name in enumerate(NUTRIENT_FEATURES):
        vec[i] = float(values[name])
    vec[-1] = float(age_years)
    return vec


def recipes_to_matrix(
    recipes: Sequence[dict[str, float]],
    ages: Iterable[float],
    ingredients: pd.DataFrame,
) -> np.ndarray:
    """Stack many encoded recipes into a 2-D (n_samples, N_FEATURES) matrix."""
    ages_list = list(ages)
    if len(ages_list) != len(recipes):
        raise ValueError("recipes and ages must have the same length")
    out = np.zeros((len(recipes), N_FEATURES), dtype=np.float32)
    for i, (r, a) in enumerate(zip(recipes, ages_list)):
        out[i] = recipe_to_vector(r, a, ingredients)
    return out
