"""Rule-based nutritional scorer.

This is the *reference* score that the ML models (D3, D4) are trained to
approximate. It is deterministic, explainable, and grounded in
data/targets/aafco_lifestages.json.

Inputs:
  - recipe: dict[ingredient_key -> grams_as_fed]
  - ingredients: pandas DataFrame indexed by 'key' with per-100g-as-fed nutrient
    columns (moisture_g, protein_g, fat_g, calcium_mg, phosphorus_mg,
    taurine_mg, magnesium_mg, sodium_mg, potassium_mg)
  - age_years: float

Output:
  ScoreReport with overall score (0-100), per-nutrient sub-scores, and the
  per-100g-DM nutrient values that produced them.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.nutrition.targets import NutrientBand, targets_for

NUTRIENT_COLUMNS = (
    "protein_g",
    "fat_g",
    "calcium_mg",
    "phosphorus_mg",
    "taurine_mg",
    "magnesium_mg",
    "sodium_mg",
    "potassium_mg",
)


@dataclass(frozen=True)
class ScoreReport:
    overall: float
    per_nutrient: dict[str, float]
    values_per_100g_dm: dict[str, float] = field(default_factory=dict)


def _band_score(value: float, band: NutrientBand) -> float:
    """100 inside [min,max], linear decay to 0 at ±50 % outside the band."""
    if band.min <= value <= band.max:
        return 100.0
    if value < band.min:
        floor = band.min * 0.5
        if value <= floor:
            return 0.0
        return 100.0 * (value - floor) / (band.min - floor)
    ceil = band.max * 1.5
    if value >= ceil:
        return 0.0
    return 100.0 * (ceil - value) / (ceil - band.max)


def per_100g_dm(recipe: dict[str, float], ingredients: pd.DataFrame) -> dict[str, float]:
    missing = [k for k in recipe if k not in ingredients.index]
    if missing:
        raise KeyError(f"unknown ingredient key(s): {missing}")
    if not recipe:
        raise ValueError("empty recipe")

    grams = pd.Series(recipe, dtype=float)
    rows = ingredients.loc[grams.index]

    moisture_g = float((grams * rows["moisture_g"] / 100.0).sum())
    total_as_fed = float(grams.sum())
    dm_g = total_as_fed - moisture_g
    if dm_g <= 0:
        raise ValueError("recipe has no dry matter; check moisture values")

    out: dict[str, float] = {}
    for col in NUTRIENT_COLUMNS:
        total = float((grams * rows[col] / 100.0).sum())
        out[col] = total / dm_g * 100.0
    out["ca_p_ratio"] = (out["calcium_mg"] / out["phosphorus_mg"]) if out["phosphorus_mg"] else 0.0
    return out


def score(recipe: dict[str, float], ingredients: pd.DataFrame, age_years: float) -> ScoreReport:
    values = per_100g_dm(recipe, ingredients)
    bands = targets_for(age_years)
    sub: dict[str, float] = {}
    for nutrient, band in bands.items():
        if nutrient not in values:
            continue
        sub[nutrient] = _band_score(values[nutrient], band)
    overall = sum(sub.values()) / len(sub) if sub else 0.0
    return ScoreReport(overall=overall, per_nutrient=sub, values_per_100g_dm=values)
