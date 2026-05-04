"""Score and rank candidate recipes for a specific cat.

The ML model takes per-100g-DM nutrient features (see ml/features.py),
so it generalises to *any* recipe — built-in shortlist ingredients and
user-added custom ingredients alike — as long as the merged
ingredients table has a nutrient row for every key in the recipe.

After scoring, the mean preference weight over the recipe's
ingredients applies a small ±points nudge (favouring foods the cat
actually eats). The blend is gentle (max ±8 points) so nutrition still
dominates ranking — a refused-but-balanced recipe still beats an
eagerly-eaten unbalanced one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.feedback.weights import DEFAULT_WEIGHT, weight_for
from src.ml.features import recipes_to_matrix

PREF_NUDGE_MAX = 8.0  # max absolute shift from preference signal


@dataclass(frozen=True)
class Ranked:
    recipe: dict[str, float]
    ml_score: float
    pref_mean: float
    blended: float


def _pref_mean(recipe: dict[str, float], weights: dict[str, float]) -> float:
    if not recipe:
        return DEFAULT_WEIGHT
    return float(np.mean([weight_for(weights, k) for k in recipe]))


def rank(
    candidates: list[dict[str, float]],
    *,
    model,
    age_years: float,
    weights: dict[str, float],
    ingredients: pd.DataFrame,
    top_n: int = 5,
) -> list[Ranked]:
    """Rank candidate recipes for a cat.

    `ingredients` must include nutrient rows for every key referenced in
    `candidates` (the merged built-in + custom table from
    `data.library.load_ingredients()` is the right thing to pass).
    """
    if not candidates:
        return []

    X = recipes_to_matrix(
        candidates, ages=[age_years] * len(candidates), ingredients=ingredients
    )
    preds = model.predict(X)

    out: list[Ranked] = []
    for recipe, ml in zip(candidates, preds):
        pm = _pref_mean(recipe, weights)
        blended = float(ml) + PREF_NUDGE_MAX * (pm - DEFAULT_WEIGHT)
        out.append(Ranked(
            recipe=recipe, ml_score=float(ml), pref_mean=pm, blended=blended,
        ))
    out.sort(key=lambda r: r.blended, reverse=True)
    return out[:top_n]
