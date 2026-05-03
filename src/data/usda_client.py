"""Thin wrapper around USDA FoodData Central with on-disk JSON cache.

Design:
  - Layer 1: per-fdcId JSON cached under data/raw/usda/{fdcId}.json. Cached
    files are reused forever (USDA records don't change for our purposes).
  - Layer 2: build_dataset.py turns the raw JSON into data/processed/ingredients.csv.
    Once that exists, no other module ever touches the network.

Set cache_only=True (default during development after D2) to refuse network
calls — useful as a safety against quota burn or accidental online usage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import requests

from src.config import RAW_DIR, usda_api_key

API_URL = "https://api.nal.usda.gov/fdc/v1/food/{fdc_id}"

# USDA nutrient IDs we extract. Mapping → our internal column names.
# Energy is handled separately because Foundation Foods records expose
# Atwater General (2047) and Atwater Specific (2048) instead of the
# SR Legacy id (1008), and our scorer wants exactly one kcal value.
NUTRIENT_MAP = {
    1003: "protein_g",
    1004: "fat_g",
    1005: "carb_g",
    1051: "moisture_g",
    1087: "calcium_mg",
    1091: "phosphorus_mg",
    1090: "magnesium_mg",
    1093: "sodium_mg",
    1092: "potassium_mg",
    1234: "taurine_mg",
}
# All output columns including the energy column resolved below.
OUTPUT_COLS = list(NUTRIENT_MAP.values()) + ["kcal"]
ENERGY_PRIORITY = [2047, 2048, 1008]  # Atwater General > Specific > legacy


@dataclass
class UsdaClient:
    cache_dir: Path = RAW_DIR
    cache_only: bool = False
    timeout: int = 15

    def __post_init__(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, fdc_id: int) -> Path:
        return self.cache_dir / f"{fdc_id}.json"

    def get_raw(self, fdc_id: int) -> dict:
        path = self._cache_path(fdc_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        if self.cache_only:
            raise RuntimeError(
                f"fdcId {fdc_id} not cached and cache_only=True. "
                "Re-run build_dataset.py with cache_only=False to fetch it."
            )
        url = API_URL.format(fdc_id=fdc_id)
        r = requests.get(url, params={"api_key": usda_api_key()}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        return data

    def get_nutrients(self, fdc_id: int) -> dict[str, float]:
        """Return our normalized per-100g nutrient dict (zeros where USDA omits)."""
        data = self.get_raw(fdc_id)
        out = {col: 0.0 for col in OUTPUT_COLS}
        energy_seen: dict[int, float] = {}
        for item in data.get("foodNutrients", []):
            nid = (item.get("nutrient") or {}).get("id") or item.get("nutrientId")
            amount = float(item.get("amount") or item.get("value") or 0.0)
            if nid in NUTRIENT_MAP:
                out[NUTRIENT_MAP[nid]] = amount
            elif nid in ENERGY_PRIORITY:
                energy_seen[nid] = amount
        for eid in ENERGY_PRIORITY:
            if eid in energy_seen:
                out["kcal"] = energy_seen[eid]
                break
        return out
