"""Per-cat per-ingredient preference weights.

The weight is a soft multiplier centred on 1.0 that nudges the
recommender toward ingredients the cat ate willingly and away from
ingredients the cat refused. Weights live in SQLite and are updated
after each feeding outcome.

  ate_all  →  +0.10 per ingredient in the recipe
  half     →  +0.00 (neutral signal)
  refused  →  -0.15 per ingredient in the recipe

Clamped to [0.30, 1.80] so a single bad meal can't kill an ingredient
forever and a single good meal can't dominate ranking.
"""
from __future__ import annotations

import sqlite3

DELTA = {"ate_all": 0.10, "half": 0.00, "refused": -0.15}
W_MIN, W_MAX = 0.30, 1.80
DEFAULT_WEIGHT = 1.0


def get_weights(conn: sqlite3.Connection, cat_id: int) -> dict[str, float]:
    """Return all stored preference weights for a cat. Missing keys default to 1.0."""
    rows = conn.execute(
        "SELECT ingredient_key, weight FROM preference_weights WHERE cat_id = ?",
        (cat_id,),
    ).fetchall()
    return {r["ingredient_key"]: float(r["weight"]) for r in rows}


def weight_for(weights: dict[str, float], ingredient_key: str) -> float:
    return weights.get(ingredient_key, DEFAULT_WEIGHT)


def update_after_feeding(
    conn: sqlite3.Connection,
    *,
    cat_id: int,
    recipe: dict[str, float],
    response: str,
) -> dict[str, float]:
    """Apply the response delta to every ingredient in the recipe.

    Inserts a row at DEFAULT_WEIGHT first time we see an ingredient for
    this cat, then nudges. Returns the updated weights dict.
    """
    if response not in DELTA:
        raise ValueError(f"unknown response {response!r}")
    delta = DELTA[response]
    weights = get_weights(conn, cat_id)
    for key in recipe:
        new_w = max(W_MIN, min(W_MAX, weights.get(key, DEFAULT_WEIGHT) + delta))
        conn.execute(
            "INSERT INTO preference_weights(cat_id, ingredient_key, weight) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(cat_id, ingredient_key) DO UPDATE SET weight = excluded.weight",
            (cat_id, key, new_w),
        )
        weights[key] = new_w
    conn.commit()
    return weights
