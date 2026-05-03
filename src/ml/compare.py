"""Side-by-side RF vs GBR comparison: evaluate both, print a table,
save a predicted-vs-actual scatter plot for the ODT report.

Run after both trainers have produced their joblib files:
    py -m src.ml.train_rf
    py -m src.ml.train_alt
    py -m src.ml.compare
"""
from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # headless — never try to open a GUI window
import matplotlib.pyplot as plt
import numpy as np

from src.config import MODELS_DIR, ROOT
from src.ml.dataset import prepare
from src.ml.evaluate import evaluate

FIGURE_PATH = ROOT / "docs" / "figures" / "rf_vs_gbr.png"


def _scatter(ax, y_true, y_pred, label: str, r2: float) -> None:
    rng = np.random.default_rng(0)
    n = len(y_true)
    if n > 800:
        idx = rng.choice(n, size=800, replace=False)
        yt, yp = y_true[idx], y_pred[idx]
    else:
        yt, yp = y_true, y_pred
    ax.scatter(yt, yp, s=8, alpha=0.4, edgecolors="none")
    lo, hi = float(min(yt.min(), yp.min())) - 2, float(max(yt.max(), yp.max())) + 2
    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1, linestyle="--", label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Rule scorer (true) overall")
    ax.set_ylabel("Model predicted overall")
    ax.set_title(f"{label}\nR² (test) = {r2:.4f}")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)


def main() -> None:
    rf_path = MODELS_DIR / "rf.joblib"
    gbr_path = MODELS_DIR / "gbr.joblib"
    for p in (rf_path, gbr_path):
        if not p.exists():
            raise SystemExit(
                f"Missing {p}. Run train_rf.py and train_alt.py first."
            )

    print("Reproducing the shared dataset (same seed/n as both trainers) …")
    ds = prepare()

    print("Loading models …")
    rf = joblib.load(rf_path)
    gbr = joblib.load(gbr_path)

    print("\nEvaluating both on the held-out test set …")
    m_rf = evaluate(rf, ds.X_train, ds.y_train, ds.X_test, ds.y_test)
    m_gbr = evaluate(gbr, ds.X_train, ds.y_train, ds.X_test, ds.y_test)

    rf_size = rf_path.stat().st_size / 1_048_576
    gbr_size = gbr_path.stat().st_size / 1_048_576

    # Pretty comparison table
    header = f"{'metric':<22}{'RandomForest':>16}{'GradientBoosting':>20}"
    print("\n" + header)
    print("-" * len(header))
    rows = [
        ("MAE",                f"{m_rf.mae:.3f}",         f"{m_gbr.mae:.3f}"),
        ("RMSE",               f"{m_rf.rmse:.3f}",        f"{m_gbr.rmse:.3f}"),
        ("R^2 (test)",         f"{m_rf.r2:.4f}",          f"{m_gbr.r2:.4f}"),
        ("R^2 (5-fold CV)",    f"{m_rf.cv_r2_mean:.4f}",  f"{m_gbr.cv_r2_mean:.4f}"),
        ("CV std",             f"+/-{m_rf.cv_r2_std:.4f}", f"+/-{m_gbr.cv_r2_std:.4f}"),
        ("model size (MB)",    f"{rf_size:.2f}",          f"{gbr_size:.2f}"),
    ]
    for name, a, b in rows:
        print(f"{name:<22}{a:>16}{b:>20}")

    winner_r2 = "RandomForest" if m_rf.r2 >= m_gbr.r2 else "GradientBoosting"
    winner_size = "RandomForest" if rf_size <= gbr_size else "GradientBoosting"
    print(f"\nHigher R^2 (test): {winner_r2}    |    Smaller model: {winner_size}")

    # Side-by-side scatter plot
    rf_pred = rf.predict(ds.X_test)
    gbr_pred = gbr.predict(ds.X_test)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    _scatter(axes[0], ds.y_test, rf_pred, "RandomForestRegressor", m_rf.r2)
    _scatter(axes[1], ds.y_test, gbr_pred, "GradientBoostingRegressor", m_gbr.r2)
    fig.suptitle("Rule scorer vs ML predictions on held-out test set", fontsize=12)
    fig.tight_layout()

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, dpi=150, bbox_inches="tight")
    print(f"\nSaved comparison plot → {FIGURE_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
