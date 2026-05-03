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
    IngredientSpec("chicken_breast",  "Chicken breast",       "protein", "soft", fdc_id=2646170),
    IngredientSpec("chicken_thigh",   "Chicken thigh",        "protein", "soft", fdc_id=173627),
    IngredientSpec("turkey_breast",   "Turkey breast",        "protein", "soft", fdc_id=171098),
    IngredientSpec("turkey_thigh",    "Turkey thigh",         "protein", "soft", fdc_id=171533),
    IngredientSpec("beef_lean",       "Lean ground beef",     "protein", "soft", fdc_id=2514744),
    IngredientSpec("pork_lean",       "Lean pork",            "protein", "soft", fdc_id=168249),
    IngredientSpec("rabbit",          "Rabbit",               "protein", "soft", fdc_id=172521),
    IngredientSpec("duck",            "Duck (no skin)",       "protein", "soft", fdc_id=172410),
    IngredientSpec("salmon",          "Salmon",               "protein", "soft", fdc_id=173686),
    IngredientSpec("sardine",         "Sardine",              "protein", "soft", fdc_id=175139),
    IngredientSpec("tuna_light",      "Light tuna",           "protein", "soft", fdc_id=171986),
    IngredientSpec("cod",             "Cod",                  "protein", "soft", fdc_id=171955),

    # --- Organs (key for taurine, vit A, B-vitamins)
    IngredientSpec("chicken_liver",   "Chicken liver",        "organ",   "soft", fdc_id=171060),
    IngredientSpec("chicken_heart",   "Chicken heart",        "organ",   "medium", fdc_id=171458),
    IngredientSpec("beef_liver",      "Beef liver",           "organ",   "soft", fdc_id=169451),
    IngredientSpec("beef_kidney",     "Beef kidney",          "organ",   "soft", fdc_id=169449),

    # --- Eggs / dairy
    IngredientSpec("egg_whole",       "Whole egg (cooked)",   "protein", "soft", fdc_id=173424),
    IngredientSpec("egg_yolk",        "Egg yolk",             "protein", "soft", fdc_id=172184),

    # --- Carbs / fiber (small amounts; cats are obligate carnivores)
    IngredientSpec("white_rice",      "Cooked white rice",    "carb",    "soft", fdc_id=168930),
    IngredientSpec("oats",            "Cooked oats",          "carb",    "soft", fdc_id=168873),
    IngredientSpec("pumpkin",         "Cooked pumpkin",       "veg",     "soft", fdc_id=168449),
    IngredientSpec("sweet_potato",    "Cooked sweet potato",  "veg",     "soft", fdc_id=168483),
    IngredientSpec("carrot",          "Cooked carrot",        "veg",     "medium", fdc_id=170394),
    IngredientSpec("zucchini",        "Cooked zucchini",      "veg",     "soft", fdc_id=169292),

    # --- Fats
    IngredientSpec("salmon_oil",      "Salmon oil",           "fat",     "soft", fdc_id=172343),
    IngredientSpec("olive_oil",       "Olive oil",            "fat",     "soft", fdc_id=171413),

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
