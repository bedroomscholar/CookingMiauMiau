"""Candidate-recipe generator.

Mirrors the per-category sampling structure used by ml/synth.py for
training, but draws only from the cat's filtered ingredient pool. We
generate many candidates and let rank.py pick the best — the ML scorer
is fast enough that 200+ candidates is fine.
"""
from __future__ import annotations

import random
from collections import defaultdict

import pandas as pd

CATEGORY_PLAN: dict[str, tuple[int, int, float, float]] = {
    "protein":    (1, 3, 30.0, 200.0),
    "organ":      (0, 2,  5.0,  50.0),
    "carb":       (0, 1,  5.0,  60.0),
    "veg":        (0, 2,  5.0,  40.0),
    "fat":        (0, 1,  1.0,  10.0),
    "supplement": (0, 3,  0.05,  2.0),
}


def _by_category(ingredients: pd.DataFrame) -> dict[str, list[str]]:
    pools: dict[str, list[str]] = defaultdict(list)
    for key, row in ingredients.iterrows():
        pools[row["category"]].append(key)
    return pools


def random_recipe(pools: dict[str, list[str]], rng: random.Random) -> dict[str, float] | None:
    """Sample one recipe. Returns None if no protein is available at all."""
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


def propose(
    ingredients: pd.DataFrame, n_candidates: int = 200, seed: int | None = None
) -> list[dict[str, float]]:
    """Generate a batch of candidate recipes from the (already filtered) pool."""
    rng = random.Random(seed)
    pools = _by_category(ingredients)
    out: list[dict[str, float]] = []
    while len(out) < n_candidates:
        r = random_recipe(pools, rng)
        if r is None:
            raise ValueError("filtered ingredient pool has no proteins available")
        out.append(r)
    return out
