"""Shared synth → label → encode → split pipeline.

Both train_rf.py and train_alt.py call `prepare()` with the same defaults
so the RF-vs-GBR comparison sees identical (X_train, y_train, X_test,
y_test). Without this factoring, a tweak to one trainer's seed/n could
silently invalidate the comparison plot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import INGREDIENTS_CSV
from src.ml.features import recipes_to_matrix
from src.ml.synth import generate
from src.nutrition.scorer import score


@dataclass(frozen=True)
class Dataset:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    ingredients: pd.DataFrame


def label(recipes, ages, ingredients) -> np.ndarray:
    y = np.zeros(len(recipes), dtype=np.float32)
    for i, (r, a) in enumerate(zip(recipes, ages)):
        y[i] = score(r, ingredients, age_years=a).overall
    return y


def prepare(n: int = 15000, seed: int = 42, test_size: float = 0.2) -> Dataset:
    ingredients = pd.read_csv(INGREDIENTS_CSV).set_index("key")
    recipes, ages = generate(n, seed=seed)
    y = label(recipes, ages, ingredients)
    X = recipes_to_matrix(recipes, ages)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    return Dataset(X_train, X_test, y_train, y_test, ingredients)
