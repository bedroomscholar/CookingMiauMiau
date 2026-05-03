"""Train the alternative model (GradientBoostingRegressor).

Same dataset preparation as train_rf.py — identical X_train, y_train,
X_test, y_test thanks to dataset.prepare(). The point is a controlled
A/B: the only thing that varies between the two trainers is the
estimator class and its hyperparameters.

Usage:
    py -m src.ml.train_alt
"""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.ensemble import GradientBoostingRegressor

from src.config import MODELS_DIR
from src.ml.dataset import prepare
from src.ml.evaluate import evaluate

MODEL_PATH = MODELS_DIR / "gbr.joblib"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=400,
                   help="boosting iterations; GBR usually needs more weak learners than RF needs trees")
    p.add_argument("--max-depth", type=int, default=4,
                   help="GBR trees should stay shallow — depth=3-5 is typical")
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--min-samples-leaf", type=int, default=5)
    args = p.parse_args()

    print(f"Preparing dataset (n={args.n}, seed={args.seed}) …")
    t0 = time.time()
    ds = prepare(n=args.n, seed=args.seed)
    print(f"  done in {time.time() - t0:.1f}s.")

    print(f"\nTraining GradientBoostingRegressor "
          f"(n_estimators={args.n_estimators}, max_depth={args.max_depth}, "
          f"learning_rate={args.learning_rate}) …")
    t0 = time.time()
    model = GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.seed,
    )
    model.fit(ds.X_train, ds.y_train)
    print(f"  fit in {time.time() - t0:.1f}s")

    print("\nEvaluating …")
    metrics = evaluate(model, ds.X_train, ds.y_train, ds.X_test, ds.y_test)
    print(metrics.pretty(label="GradientBoostingRegressor"))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1_048_576
    print(f"\nSaved {MODEL_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
