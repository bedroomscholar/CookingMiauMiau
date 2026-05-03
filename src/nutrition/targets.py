"""Life-stage-aware nutrient targets.

The user enters a date of birth; the system computes age in years and picks the
matching life stage from data/targets/aafco_lifestages.json. The scorer then
compares a recipe's per-100g-DM nutrient values against that stage's targets.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

from src.config import AAFCO_TARGETS


@dataclass(frozen=True)
class NutrientBand:
    min: float
    target: float
    max: float
    unit: str


def compute_age_years(dob: date, today: date | None = None) -> float:
    """Age in years as a float (e.g. 10.42). Handles leap years correctly."""
    today = today or date.today()
    if dob > today:
        raise ValueError(f"date of birth {dob} is in the future relative to {today}")
    years = today.year - dob.year
    try:
        anniversary = dob.replace(year=today.year)
    except ValueError:
        anniversary = dob.replace(year=today.year, day=28)
    if today < anniversary:
        years -= 1
        prev_anniv = anniversary.replace(year=today.year - 1)
        next_anniv = anniversary
    else:
        prev_anniv = anniversary
        try:
            next_anniv = anniversary.replace(year=today.year + 1)
        except ValueError:
            next_anniv = anniversary.replace(year=today.year + 1, day=28)
    fraction = (today - prev_anniv).days / max((next_anniv - prev_anniv).days, 1)
    return years + fraction


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(AAFCO_TARGETS, "r", encoding="utf-8") as f:
        return json.load(f)


def stage_for(age_years: float) -> str:
    boundaries = _load()["_about"]["stage_boundaries_years"]
    for stage, (lo, hi) in boundaries.items():
        if lo <= age_years < hi:
            return stage
    raise ValueError(f"no stage matches age {age_years}")


def targets_for(age_years: float) -> dict[str, NutrientBand]:
    stage = stage_for(age_years)
    raw = _load()["stages"][stage]
    return {
        name: NutrientBand(min=v["min"], target=v["target"], max=v["max"], unit=v["unit"])
        for name, v in raw.items()
    }


def targets_for_dob(dob: date, today: date | None = None) -> tuple[str, dict[str, NutrientBand]]:
    age = compute_age_years(dob, today)
    return stage_for(age), targets_for(age)
