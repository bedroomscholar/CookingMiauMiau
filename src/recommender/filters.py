"""Cat-aware ingredient filtering.

Two hard filters today:
  - Taboos: explicit ingredient_keys the cat must never receive.
  - Texture: a cat with bad teeth (texture_max='soft') can't have
    medium- or hard-textured items. Pairs are: hard >= medium >= soft.

Filters are applied to the canonical ingredients DataFrame; the
generator and ranker only ever see what the cat is allowed to eat.
"""
from __future__ import annotations

from typing import Iterable

import pandas as pd

TEXTURE_RANK = {"soft": 0, "medium": 1, "hard": 2}


def applicable_ingredients(
    ingredients: pd.DataFrame,
    *,
    taboos: Iterable[str] = (),
    texture_max: str = "hard",
) -> pd.DataFrame:
    if texture_max not in TEXTURE_RANK:
        raise ValueError(f"texture_max must be one of {tuple(TEXTURE_RANK)}")
    max_rank = TEXTURE_RANK[texture_max]

    taboo_set = set(taboos)
    keep_mask = ~ingredients.index.isin(taboo_set)
    if "texture" in ingredients.columns:
        rank_series = ingredients["texture"].map(TEXTURE_RANK).fillna(max_rank).astype(int)
        keep_mask &= rank_series <= max_rank
    return ingredients[keep_mask].copy()
