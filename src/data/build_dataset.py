"""One-shot script: pull each shortlist ingredient from USDA (or cache),
apply manual overrides for supplements/taurine gaps, and write
data/processed/ingredients.csv.

Run once, after which the app is fully offline:
    py -m src.data.build_dataset
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.config import INGREDIENTS_CSV, PROCESSED_DIR
from src.data.ingredient_shortlist import SHORTLIST
from src.data.usda_client import OUTPUT_COLS, UsdaClient

# Fallbacks for nutrients USDA often omits (mg per 100g as-fed) and for
# pure supplements that aren't USDA foods at all.
MANUAL_OVERRIDES: dict[str, dict[str, float]] = {
    # Taurine values from published feline-nutrition references (Spitze 2003,
    # Hedberg 2007). USDA's database mostly lacks taurine.
    "chicken_breast":  {"taurine_mg": 18.0},
    "chicken_thigh":   {"taurine_mg": 35.0},
    "chicken_liver":   {"taurine_mg": 110.0},
    "chicken_heart":   {"taurine_mg": 118.0},
    "turkey_breast":   {"taurine_mg": 30.0},
    "turkey_thigh":    {"taurine_mg": 60.0},
    "beef_lean":       {"taurine_mg": 36.0},
    "beef_liver":      {"taurine_mg": 19.0},
    "beef_kidney":     {"taurine_mg": 22.0},
    "pork_lean":       {"taurine_mg": 50.0},
    "rabbit":          {"taurine_mg": 37.0},
    "duck":            {"taurine_mg": 30.0},
    "salmon":          {"taurine_mg": 130.0},
    "sardine":         {"taurine_mg": 147.0},
    "tuna_light":      {"taurine_mg": 70.0},
    "cod":             {"taurine_mg": 31.0},
    "egg_whole":       {"taurine_mg": 0.0},
    "egg_yolk":        {"taurine_mg": 0.0},

    # Pure supplements: not in USDA. Defined entirely manually.
    "taurine_powder": {
        "moisture_g": 0.0, "kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carb_g": 0.0,
        "calcium_mg": 0.0, "phosphorus_mg": 0.0, "magnesium_mg": 0.0,
        "sodium_mg": 0.0, "potassium_mg": 0.0, "taurine_mg": 100_000.0,
    },
    "calcium_carb": {
        "moisture_g": 0.0, "kcal": 0.0, "protein_g": 0.0, "fat_g": 0.0, "carb_g": 0.0,
        "calcium_mg": 40_000.0, "phosphorus_mg": 0.0, "magnesium_mg": 0.0,
        "sodium_mg": 0.0, "potassium_mg": 0.0, "taurine_mg": 0.0,
    },
    "eggshell_powder": {
        "moisture_g": 0.0, "kcal": 0.0, "protein_g": 4.0, "fat_g": 0.0, "carb_g": 0.0,
        "calcium_mg": 38_000.0, "phosphorus_mg": 120.0, "magnesium_mg": 400.0,
        "sodium_mg": 130.0, "potassium_mg": 80.0, "taurine_mg": 0.0,
    },
    "kelp_powder": {
        "moisture_g": 6.0, "kcal": 43.0, "protein_g": 1.7, "fat_g": 0.6, "carb_g": 9.6,
        "calcium_mg": 168.0, "phosphorus_mg": 42.0, "magnesium_mg": 121.0,
        "sodium_mg": 233.0, "potassium_mg": 89.0, "taurine_mg": 0.0,
    },
    "brewers_yeast": {
        "moisture_g": 6.0, "kcal": 295.0, "protein_g": 38.8, "fat_g": 1.8, "carb_g": 41.2,
        "calcium_mg": 30.0, "phosphorus_mg": 1900.0, "magnesium_mg": 230.0,
        "sodium_mg": 51.0, "potassium_mg": 1888.0, "taurine_mg": 0.0,
    },
}


def build(cache_only: bool = False) -> pd.DataFrame:
    client = UsdaClient(cache_only=cache_only)
    rows = []
    for spec in SHORTLIST:
        if spec.fdc_id == 0 and spec.key not in MANUAL_OVERRIDES:
            print(f"  SKIP {spec.key}: no fdc_id and no manual override", file=sys.stderr)
            continue

        row = {col: 0.0 for col in OUTPUT_COLS}
        if spec.fdc_id != 0:
            row.update(client.get_nutrients(spec.fdc_id))
        if spec.key in MANUAL_OVERRIDES:
            row.update(MANUAL_OVERRIDES[spec.key])

        row["key"] = spec.key
        row["display"] = spec.display
        row["category"] = spec.category
        row["texture"] = spec.texture
        row["fdc_id"] = spec.fdc_id
        rows.append(row)
        print(f"  OK  {spec.key:<18} {row['protein_g']:6.1f}g protein  {row['kcal']:5.0f} kcal")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cols = ["key", "display", "category", "texture", "fdc_id"] + list(OUTPUT_COLS)
    df = df[cols].set_index("key")
    df.to_csv(INGREDIENTS_CSV)
    print(f"\nWrote {len(df)} ingredients → {INGREDIENTS_CSV}")
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache-only", action="store_true",
                   help="Refuse network calls; use only data/raw/usda/*.json")
    args = p.parse_args()
    build(cache_only=args.cache_only)


if __name__ == "__main__":
    main()
