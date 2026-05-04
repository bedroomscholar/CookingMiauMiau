"""User-extensible ingredient library.

Two CSVs at runtime:
  - data/processed/ingredients.csv          — frozen, built by build_dataset.py
  - data/processed/custom_ingredients.csv   — appended to via the GUI

`load_ingredients()` returns a merged DataFrame with an extra boolean
column `custom` so the UI can mark user rows. The ML training pipeline
intentionally trains on the base CSV only (custom rows are user-local
and not part of the released model), but inference works on any
ingredient — the trained model takes per-100g-DM nutrient features
(see ml/features.py), so a custom ingredient with a valid nutrient row
is scored exactly like a built-in one.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.config import INGREDIENTS_CSV, PROCESSED_DIR

CUSTOM_CSV = PROCESSED_DIR / "custom_ingredients.csv"

VALID_CATEGORIES = ("protein", "organ", "carb", "veg", "fat", "supplement")
VALID_TEXTURES = ("soft", "medium", "hard")

NUTRIENT_COLUMNS = (
    "protein_g", "fat_g", "carb_g", "moisture_g",
    "calcium_mg", "phosphorus_mg", "magnesium_mg",
    "sodium_mg", "potassium_mg", "taurine_mg", "kcal",
)
ALL_COLUMNS = ("key", "display", "category", "texture", "fdc_id") + NUTRIENT_COLUMNS

_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")


def _read_or_empty(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path).set_index("key")
    return pd.DataFrame(columns=[c for c in ALL_COLUMNS if c != "key"]).rename_axis("key")


def load_ingredients() -> pd.DataFrame:
    base = pd.read_csv(INGREDIENTS_CSV).set_index("key").assign(custom=False)
    custom = _read_or_empty(CUSTOM_CSV).assign(custom=True)
    if custom.empty:
        return base
    # On any accidental key collision the user's row wins so they can override.
    merged = pd.concat([base[~base.index.isin(custom.index)], custom])
    return merged


def list_custom_keys() -> set[str]:
    return set(_read_or_empty(CUSTOM_CSV).index)


def add_custom(
    *,
    key: str,
    display: str,
    category: str,
    texture: str,
    nutrients: dict[str, float],
    fdc_id: int = 0,
) -> None:
    """Validate inputs and append a row to custom_ingredients.csv."""
    if not _KEY_RE.match(key):
        raise ValueError(
            f"key must be lowercase letters/digits/underscores starting with a letter: {key!r}"
        )
    if category not in VALID_CATEGORIES:
        raise ValueError(f"category must be one of {VALID_CATEGORIES}")
    if texture not in VALID_TEXTURES:
        raise ValueError(f"texture must be one of {VALID_TEXTURES}")
    if not display.strip():
        raise ValueError("display name is required")

    base_keys = set(pd.read_csv(INGREDIENTS_CSV)["key"])
    if key in base_keys:
        raise ValueError(f"key {key!r} already exists in the built-in catalogue")
    if key in list_custom_keys():
        raise ValueError(f"key {key!r} already exists as a custom ingredient")

    row = {c: 0.0 for c in NUTRIENT_COLUMNS}
    for k, v in nutrients.items():
        if k not in NUTRIENT_COLUMNS:
            raise ValueError(f"unknown nutrient column: {k!r}")
        v = float(v)
        if v < 0:
            raise ValueError(f"{k} must be non-negative (got {v})")
        row[k] = v
    row.update({
        "key": key, "display": display.strip(),
        "category": category, "texture": texture, "fdc_id": int(fdc_id),
    })

    df = _read_or_empty(CUSTOM_CSV).reset_index()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df = df[list(ALL_COLUMNS)]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CUSTOM_CSV, index=False)


def delete_custom(key: str) -> None:
    df = _read_or_empty(CUSTOM_CSV)
    if key not in df.index:
        raise KeyError(f"no custom ingredient {key!r}")
    df.drop(index=key).to_csv(CUSTOM_CSV)
