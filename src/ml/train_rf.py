"""Train the primary model (RandomForestRegressor).

Uses the shared `dataset.prepare()` pipeline so the train/test split is
identical to what train_alt.py sees — that's required for the
RF-vs-GBR comparison plot in D4 to be a fair apples-to-apples readout.

Usage:
    py -m src.ml.train_rf
    py -m src.ml.train_rf --n 8000 --seed 7
"""
from __future__ import annotations

import argparse
import time

import joblib
from sklearn.ensemble import RandomForestRegressor

from src.config import MODELS_DIR
from src.ml.dataset import prepare
from src.ml.evaluate import evaluate

MODEL_PATH = MODELS_DIR / "rf.joblib"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=15000, help="number of synthetic samples")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=100)
    p.add_argument("--max-depth", type=int, default=16,
                   help="caps tree depth so the joblib stays small")
    p.add_argument("--min-samples-leaf", type=int, default=2,
                   help="prunes the trees further; bigger value = smaller model")
    args = p.parse_args()

    print(f"Preparing dataset (n={args.n}, seed={args.seed}) …")
    t0 = time.time()
    ds = prepare(n=args.n, seed=args.seed)
    print(f"  done in {time.time() - t0:.1f}s.  "
          f"y range [{ds.y_train.min():.1f}, {ds.y_train.max():.1f}] "
          f"mean {ds.y_train.mean():.1f}")

    print(f"\nTraining RandomForestRegressor "
          f"(n_estimators={args.n_estimators}, max_depth={args.max_depth}) …")
    t0 = time.time()
    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        n_jobs=-1,
        random_state=args.seed,
    )
    model.fit(ds.X_train, ds.y_train)
    print(f"  fit in {time.time() - t0:.1f}s")

    print("\nEvaluating …")
    metrics = evaluate(model, ds.X_train, ds.y_train, ds.X_test, ds.y_test)
    print(metrics.pretty(label="RandomForestRegressor"))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    size_mb = MODEL_PATH.stat().st_size / 1_048_576
    print(f"\nSaved {MODEL_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
