"""Synthesize random but plausible cat-food recipes for ML training.

Each sample is (recipe_dict, age_years) where:
  - age is uniform on [0.3, 16.0] so the model sees every life stage
  - recipe is a small mix sampled by category, with grams drawn from
    realistic per-category bands. We require >=1 protein source so the
    rule scorer never crashes on a zero-dry-matter recipe.

The random distributions don't have to be physiologically perfect — we
only need enough diversity that the rule-based scorer produces a wide
range of overall scores (50 .. 95 typically), giving the ML model a
non-trivial regression target.
"""
from __future__ import annotations

import random
from collections import defaultdict

from src.data.ingredient_shortlist import SHORTLIST

# (n_min, n_max, g_min, g_max) per category — how many items to draw and
# how many grams each gets. Supplements are dosed in tenths of a gram.
CATEGORY_PLAN: dict[str, tuple[int, int, float, float]] = {
    "protein":    (1, 3, 30.0, 200.0),
    "organ":      (0, 2,  5.0,  50.0),
    "carb":       (0, 1,  5.0,  60.0),
    "veg":        (0, 2,  5.0,  40.0),
    "fat":        (0, 1,  1.0,  10.0),
    "supplement": (0, 3,  0.05,  2.0),
}

_BY_CATEGORY: dict[str, list[str]] = defaultdict(list)
for s in SHORTLIST:
    _BY_CATEGORY[s.category].append(s.key)


def random_recipe(rng: random.Random) -> dict[str, float]:
    recipe: dict[str, float] = {}
    for cat, (nmin, nmax, gmin, gmax) in CATEGORY_PLAN.items():
        pool = _BY_CATEGORY.get(cat, [])
        if not pool:
            continue
        k = rng.randint(nmin, min(nmax, len(pool)))
        if k == 0:
            continue
        for key in rng.sample(pool, k):
            grams = rng.uniform(gmin, gmax)
            recipe[key] = round(grams, 2)
    if not any(s.key in recipe and s.category == "protein" for s in SHORTLIST):
        # Belt-and-braces: CATEGORY_PLAN forces >=1 protein, but in case
        # the protein pool is empty we'd otherwise yield an unscoreable recipe.
        key = rng.choice(_BY_CATEGORY["protein"])
        recipe[key] = round(rng.uniform(50.0, 150.0), 2)
    return recipe


def random_age(rng: random.Random) -> float:
    return round(rng.uniform(0.3, 16.0), 2)


def generate(n_samples: int, seed: int = 42) -> tuple[list[dict[str, float]], list[float]]:
    rng = random.Random(seed)
    recipes = [random_recipe(rng) for _ in range(n_samples)]
    ages = [random_age(rng) for _ in range(n_samples)]
    return recipes, ages
