from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "usda"
PROCESSED_DIR = DATA_DIR / "processed"
TARGETS_DIR = DATA_DIR / "targets"
MODELS_DIR = ROOT / "models"
DB_PATH = ROOT / "catfood.db"

INGREDIENTS_CSV = PROCESSED_DIR / "ingredients.csv"
AAFCO_TARGETS = TARGETS_DIR / "aafco_lifestages.json"

load_dotenv(ROOT / ".env")


def usda_api_key() -> str:
    key = os.environ.get("USDA_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "USDA_API_KEY missing. Set it in .env (see .env.example). "
            "Only build_dataset.py needs this; the app itself runs offline."
        )
    return key
