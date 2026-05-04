"""Candidate-recipe generator (one day's worth of food per recipe).

Each recipe represents the cat's *daily* portion, sized by body weight
and activity level via `daily_grams()`. We sample random ingredient
proportions in modest per-category bands, then scale every candidate
so its total weight matches the daily target.

Why scale rather than draw directly at target size: scaling preserves
the random ingredient *ratios* — which is what the rule scorer and the
nutrient-vector ML model both consume — without forcing the
generator's per-category bands to track arbitrary daily totals.
"""
from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

# (n_min, n_max, g_min, g_max). These are *pre-scale* bands — chosen so
# the un-scaled total is a sensible "meal-sized" 60-150g, which makes the
# scale factor close to 1-3x for a typical cat. The absolute numbers
# don't reach the user; they only set the relative mix.
CATEGORY_PLAN: dict[str, tuple[int, int, float, float]] = {
    "protein":    (1, 2, 30.0, 80.0),
    "organ":      (0, 1,  5.0, 15.0),
    "carb":       (0, 1,  5.0, 20.0),
    "veg":        (0, 1,  5.0, 15.0),
    "fat":        (0, 1,  1.0,  4.0),
    "supplement": (0, 2,  0.05, 1.0),
}

# Daily food as a fraction of body weight, by activity tier.
ACTIVITY_DAILY_FRACTION: dict[str, float] = {
    "low":    0.025,
    "medium": 0.035,
    "high":   0.045,
}


def daily_grams(weight_kg: float, activity: str) -> float:
    """Daily fresh-food target in grams for a cat at given weight & activity.

    Uses the rule-of-thumb 2.5–4.5 % of body weight per day, picked from
    `ACTIVITY_DAILY_FRACTION`. Falls back to medium if the activity
    string is unrecognised.
    """
    frac = ACTIVITY_DAILY_FRACTION.get(activity, ACTIVITY_DAILY_FRACTION["medium"])
    return float(weight_kg) * 1000.0 * frac


def _by_category(ingredients: pd.DataFrame) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    for key, row in ingredients.iterrows():
        pools[row["category"]].append(key)
    return pools


def random_recipe(pools: dict[str, list[str]], rng: random.Random) -> dict[str, float] | None:
    """Sample one recipe at meal-scale proportions. Returns None if no
    protein is available at all."""
    if not pools.get("protein"):
        return None
    recipe: dict[str, float] = {}
    for cat, (nmin, nmax, gmin, gmax) in CATEGORY_PLAN.items():
        pool = pools.get(cat) or []
        if not pool:
            continue
        k = rng.randint(nmin, min(nmax, len(pool)))
        if k == 0:
            continue
        for key in rng.sample(pool, k):
            recipe[key] = round(rng.uniform(gmin, gmax), 2)
    return recipe


def _scale_to_target(recipe: dict[str, float], target_grams: float) -> dict[str, float]:
    """Proportionally scale every gram in the recipe so the total is
    exactly target_grams. Preserves ratios — the per-100g-DM nutrient
    profile (and therefore the ML score) is unchanged."""
    total = sum(recipe.values())
    if total <= 0 or target_grams <= 0:
        return recipe
    factor = target_grams / total
    return {k: round(v * factor, 2) for k, v in recipe.items()}


def propose(
    ingredients: pd.DataFrame,
    n_candidates: int = 200,
    seed: int | None = None,
    target_grams: float | None = None,
) -> list[dict[str, float]]:
    """Generate a batch of candidate daily recipes from a (filtered) pool.

    If `target_grams` is given, every returned recipe is scaled so its
    total weight equals that target.
    """
    rng = random.Random(seed)
    pools = _by_category(ingredients)
    out: list[dict[str, float]] = []
    while len(out) < n_candidates:
        r = random_recipe(pools, rng)
        if r is None:
            raise ValueError("filtered ingredient pool has no proteins available")
        if target_grams is not None:
            r = _scale_to_target(r, target_grams)
        out.append(r)
    return out
