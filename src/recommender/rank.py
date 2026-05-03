"""Score and rank candidate recipes for a specific cat.

Scoring path picks itself based on the recipe's contents:
  - All-canonical recipe (every ingredient_key is in the trained
    SHORTLIST) → fast vectorized ML prediction.
  - Recipe contains a user-added custom ingredient → rule scorer is
    used instead, since the ML model never saw that key during
    training and faking a prediction would be dishonest.

After scoring, the mean preference weight over the recipe's
ingredients applies a small ±points nudge (favouring foods the cat
actually eats). The blend is gentle (max ±8 points) so nutrition still
dominates ranking — a refused-but-balanced recipe still beats an
eagerly-eaten unbalanced one.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from src.feedback.weights import DEFAULT_WEIGHT, weight_for
from src.ml.features import KEY_INDEX, recipes_to_matrix
from src.nutrition.scorer import score as rule_score

PREF_NUDGE_MAX = 8.0  # max absolute shift from preference signal
CANONICAL_KEYS = frozenset(KEY_INDEX)


@dataclass(frozen=True)
class Ranked:
    recipe: dict[str, float]
    ml_score: float
    pref_mean: float
    blended: float
    used_rule_scorer: bool = False


def _pref_mean(recipe: dict[str, float], weights: dict[str, float]) -> float:
    if not recipe:
        return DEFAULT_WEIGHT
    return float(np.mean([weight_for(weights, k) for k in recipe]))


def _is_canonical(recipe: dict[str, float]) -> bool:
    return CANONICAL_KEYS.issuperset(recipe)


def rank(
    candidates: list[dict[str, float]],
    *,
    model,
    age_years: float,
    weights: dict[str, float],
    top_n: int = 5,
    ingredients: Optional[pd.DataFrame] = None,
) -> list[Ranked]:
    """Rank candidate recipes for a cat.

    `ingredients` is required when any candidate may contain custom
    (non-SHORTLIST) ingredient keys — the rule scorer needs the full
    nutrient table for those.
    """
    if not candidates:
        return []

    canonical_idx: list[int] = []
    canonical_recipes: list[dict[str, float]] = []
    for i, r in enumerate(candidates):
        if _is_canonical(r):
            canonical_idx.append(i)
            canonical_recipes.append(r)

    ml_scores: dict[int, float] = {}
    if canonical_recipes:
        X = recipes_to_matrix(canonical_recipes, ages=[age_years] * len(canonical_recipes))
        for idx, pred in zip(canonical_idx, model.predict(X)):
            ml_scores[idx] = float(pred)

    out: list[Ranked] = []
    for i, recipe in enumerate(candidates):
        if i in ml_scores:
            base_score = ml_scores[i]
            used_rule = False
        else:
            if ingredients is None:
                raise ValueError(
                    "candidate contains a non-SHORTLIST key but no `ingredients` "
                    "DataFrame was passed for fallback rule scoring"
                )
            base_score = float(rule_score(recipe, ingredients, age_years=age_years).overall)
            used_rule = True

        pm = _pref_mean(recipe, weights)
        blended = base_score + PREF_NUDGE_MAX * (pm - DEFAULT_WEIGHT)
        out.append(Ranked(
            recipe=recipe, ml_score=base_score, pref_mean=pm,
            blended=blended, used_rule_scorer=used_rule,
        ))
    out.sort(key=lambda r: r.blended, reverse=True)
    return out[:top_n]
