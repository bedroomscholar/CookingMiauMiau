"""Shared regression metrics + a small pretty-printer.

Used by both train_rf.py (D3) and train_alt.py (D4) so the comparison
in the ODT report uses identical math on identical splits.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score


@dataclass(frozen=True)
class Metrics:
    mae: float
    rmse: float
    r2: float
    cv_r2_mean: float
    cv_r2_std: float

    def pretty(self, label: str = "") -> str:
        head = f"  {label}\n" if label else ""
        return (
            f"{head}    MAE        {self.mae:7.3f}\n"
            f"    RMSE       {self.rmse:7.3f}\n"
            f"    R^2 (test) {self.r2:7.4f}\n"
            f"    R^2 (5-fold CV)  {self.cv_r2_mean:.4f}  ±{self.cv_r2_std:.4f}"
        )


def evaluate(model, X_train, y_train, X_test, y_test, *, k_folds: int = 5) -> Metrics:
    """Fit-then-evaluate. Caller is expected to pre-split."""
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    r2 = r2_score(y_test, pred)
    cv = cross_val_score(
        model, X_train, y_train,
        cv=KFold(n_splits=k_folds, shuffle=True, random_state=0),
        scoring="r2",
    )
    return Metrics(
        mae=float(mae),
        rmse=float(rmse),
        r2=float(r2),
        cv_r2_mean=float(cv.mean()),
        cv_r2_std=float(cv.std()),
    )
