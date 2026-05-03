"""Hard-coded shortlist of ~30 cat-safe ingredients to fetch from USDA FoodData Central.

Each entry has:
  - key: stable internal ID used everywhere (DB, recipes, models)
  - display: human label shown in GUI
  - category: protein / organ / carb / fat / supplement / veg
  - texture: soft | medium | hard  (Changcheng's bad-teeth filter excludes "hard")
  - fdc_id: USDA FoodData Central ID, filled in on D2 by running build_dataset.py
            (left as 0 here so this file can be reviewed before any API call)
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IngredientSpec:
    key: str
    display: str
    category: str
    texture: str
    fdc_id: int = 0


SHORTLIST: list[IngredientSpec] = [
    # --- Muscle meat (proteins)
    IngredientSpec("chicken_breast",  "Chicken breast",       "protein", "soft"),
    IngredientSpec("chicken_thigh",   "Chicken thigh",        "protein", "soft"),
    IngredientSpec("turkey_breast",   "Turkey breast",        "protein", "soft"),
    IngredientSpec("turkey_thigh",    "Turkey thigh",         "protein", "soft"),
    IngredientSpec("beef_lean",       "Lean ground beef",     "protein", "soft"),
    IngredientSpec("pork_lean",       "Lean pork",            "protein", "soft"),
    IngredientSpec("rabbit",          "Rabbit",               "protein", "soft"),
    IngredientSpec("duck",            "Duck (no skin)",       "protein", "soft"),
    IngredientSpec("salmon",          "Salmon",               "protein", "soft"),
    IngredientSpec("sardine",         "Sardine",              "protein", "soft"),
    IngredientSpec("tuna_light",      "Light tuna",           "protein", "soft"),
    IngredientSpec("cod",             "Cod",                  "protein", "soft"),

    # --- Organs (key for taurine, vit A, B-vitamins)
    IngredientSpec("chicken_liver",   "Chicken liver",        "organ",   "soft"),
    IngredientSpec("chicken_heart",   "Chicken heart",        "organ",   "medium"),
    IngredientSpec("beef_liver",      "Beef liver",           "organ",   "soft"),
    IngredientSpec("beef_kidney",     "Beef kidney",          "organ",   "soft"),

    # --- Eggs / dairy
    IngredientSpec("egg_whole",       "Whole egg (cooked)",   "protein", "soft"),
    IngredientSpec("egg_yolk",        "Egg yolk",             "protein", "soft"),

    # --- Carbs / fiber (small amounts; cats are obligate carnivores)
    IngredientSpec("white_rice",      "Cooked white rice",    "carb",    "soft"),
    IngredientSpec("oats",            "Cooked oats",          "carb",    "soft"),
    IngredientSpec("pumpkin",         "Cooked pumpkin",       "veg",     "soft"),
    IngredientSpec("sweet_potato",    "Cooked sweet potato",  "veg",     "soft"),
    IngredientSpec("carrot",          "Cooked carrot",        "veg",     "medium"),
    IngredientSpec("zucchini",        "Cooked zucchini",      "veg",     "soft"),

    # --- Fats
    IngredientSpec("salmon_oil",      "Salmon oil",           "fat",     "soft"),
    IngredientSpec("olive_oil",       "Olive oil",            "fat",     "soft"),

    # --- Supplements (dosed in mg, not bulk grams; treated specially in scorer)
    IngredientSpec("taurine_powder",  "Taurine powder",       "supplement", "soft"),
    IngredientSpec("calcium_carb",    "Calcium carbonate",    "supplement", "soft"),
    IngredientSpec("eggshell_powder", "Eggshell powder",      "supplement", "medium"),
    IngredientSpec("kelp_powder",     "Kelp powder",          "supplement", "soft"),
    IngredientSpec("brewers_yeast",   "Brewer's yeast",       "supplement", "soft"),
]


def by_key(key: str) -> IngredientSpec:
    for item in SHORTLIST:
        if item.key == key:
            return item
    raise KeyError(key)
