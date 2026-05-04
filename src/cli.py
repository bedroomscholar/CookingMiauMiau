"""Command-line interface for the cat food recipe system.

Subcommands:
  init          create the SQLite schema (idempotent)
  add-cat       insert a cat profile
  list-cats     show all stored cats
  generate      produce top-N recipes for a cat
  feedback      record how the cat reacted to a served recipe
  history       last N feedings for a cat

The GUI in D6 wraps these same calls. Keeping the heavy lifting in
plain modules (db, weights, recommender) means the GUI is a thin shell.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import joblib
import pandas as pd

from src.config import INGREDIENTS_CSV, MODELS_DIR, ROOT
from src.data.library import load_ingredients
from src.feedback.db import (
    Cat, add_cat, connect, feeding_history, get_cat,
    list_cats, record_feeding,
)
from src.feedback.weights import get_weights, update_after_feeding
from src.nutrition.targets import compute_age_years
from src.recommender.filters import applicable_ingredients
from src.recommender.generator import daily_grams, propose
from src.recommender.rank import Ranked, rank

MODEL_PATH = MODELS_DIR / "gbr.joblib"
SESSIONS_DIR = ROOT / "data" / "sessions"


# --- shared helpers ----------------------------------------------------

def _load_model():
    if not MODEL_PATH.exists():
        raise SystemExit(
            f"Model not found at {MODEL_PATH}. Run `py -m src.ml.train_alt` first."
        )
    return joblib.load(MODEL_PATH)


def _load_ingredients() -> pd.DataFrame:
    if not INGREDIENTS_CSV.exists():
        raise SystemExit(
            f"Ingredients CSV not found at {INGREDIENTS_CSV}. "
            "Run `py -m src.data.build_dataset` first."
        )
    return load_ingredients()


def _session_path(cat_id: int) -> Path:
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    return SESSIONS_DIR / f"last_{cat_id}.json"


def _format_recipe(recipe: dict[str, float], ingredients: pd.DataFrame) -> str:
    parts = []
    for k, g in sorted(recipe.items(), key=lambda kv: -kv[1]):
        display = ingredients.loc[k, "display"] if k in ingredients.index else k
        parts.append(f"{display} {g:.1f}g")
    return " · ".join(parts)


# --- subcommand handlers -----------------------------------------------

def cmd_init(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    conn.close()
    print(f"DB ready at {Path(args.db).resolve() if args.db else 'default path'}")
    return 0


def cmd_add_cat(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    taboos = [t.strip() for t in (args.taboo or "").split(",") if t.strip()]
    cat_id = add_cat(
        conn,
        name=args.name,
        dob=args.dob,
        weight_kg=args.weight,
        activity=args.activity,
        texture_max=args.texture_max,
        notes=args.notes or "",
        taboos=taboos,
    )
    print(f"Added cat #{cat_id} {args.name!r} (taboos: {taboos or 'none'})")
    return 0


def cmd_list_cats(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    cats = list_cats(conn)
    if not cats:
        print("(no cats yet — try `add-cat`)")
        return 0
    today = date.today()
    for c in cats:
        age = compute_age_years(date.fromisoformat(c.dob), today)
        print(f"#{c.id} {c.name}  age {age:.1f}y  {c.weight_kg}kg  "
              f"activity={c.activity}  texture_max={c.texture_max}  "
              f"taboos={list(c.taboos) or 'none'}")
    return 0


def _generate_for_cat(
    cat: Cat, *, top_n: int, seed: int | None, conn, ingredients, model
) -> list[Ranked]:
    age = compute_age_years(date.fromisoformat(cat.dob))
    pool = applicable_ingredients(
        ingredients, taboos=cat.taboos, texture_max=cat.texture_max
    )
    target = daily_grams(cat.weight_kg, cat.activity)
    candidates = propose(pool, n_candidates=300, seed=seed, target_grams=target)
    weights = get_weights(conn, cat.id)
    return rank(
        candidates, model=model, age_years=age, weights=weights,
        top_n=top_n, ingredients=ingredients,
    )


def cmd_generate(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    cat = get_cat(conn, args.cat)
    ingredients = _load_ingredients()
    model = _load_model()

    ranked = _generate_for_cat(
        cat, top_n=args.top, seed=args.seed,
        conn=conn, ingredients=ingredients, model=model,
    )
    if not ranked:
        print("No recipes could be generated — check the cat's taboos/texture filter.")
        return 1

    age = compute_age_years(date.fromisoformat(cat.dob))
    target = daily_grams(cat.weight_kg, cat.activity)
    print(f"Top {len(ranked)} daily recipes for {cat.name} "
          f"(age {age:.1f}y, ~{target:.0f}g/day):\n")
    for i, r in enumerate(ranked, 1):
        print(f"  [{i}] blended {r.blended:5.1f}  ml {r.ml_score:5.1f}  "
              f"pref x{r.pref_mean:.2f}")
        print(f"      {_format_recipe(r.recipe, ingredients)}")

    # Cache for `feedback --pick N`
    payload = [{"recipe": r.recipe, "ml_score": r.ml_score,
                "blended": r.blended} for r in ranked]
    _session_path(cat.id).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    cat = get_cat(conn, args.cat)

    if args.recipe:
        recipe = json.loads(args.recipe)
    else:
        sess_path = _session_path(cat.id)
        if not sess_path.exists():
            raise SystemExit(
                f"No cached session for {cat.name}. Run `generate --cat {cat.name}` first, "
                "or pass --recipe '{...}' explicitly."
            )
        cached = json.loads(sess_path.read_text(encoding="utf-8"))
        if args.pick < 1 or args.pick > len(cached):
            raise SystemExit(
                f"--pick {args.pick} out of range (1..{len(cached)})."
            )
        recipe = cached[args.pick - 1]["recipe"]

    feeding_id = record_feeding(
        conn, cat_id=cat.id, recipe=recipe, response=args.response
    )
    new_weights = update_after_feeding(
        conn, cat_id=cat.id, recipe=recipe, response=args.response
    )
    changed = {k: round(new_weights[k], 3) for k in recipe}
    print(f"Recorded feeding #{feeding_id} for {cat.name}: {args.response}")
    print(f"Updated weights: {changed}")
    return 0


def cmd_history(args: argparse.Namespace) -> int:
    conn = connect(args.db)
    cat = get_cat(conn, args.cat)
    ingredients = _load_ingredients()
    rows = feeding_history(conn, cat.id, limit=args.limit)
    if not rows:
        print(f"(no feedings yet for {cat.name})")
        return 0
    for r in rows:
        recipe_str = _format_recipe(r["recipe"], ingredients)
        print(f"#{r['id']}  {r['served_at']}  {r['response']:<8}  {recipe_str}")
    return 0


# --- argument parsing --------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="catfood")
    p.add_argument("--db", default=None, help="path to SQLite DB (default from config)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="create the schema").set_defaults(func=cmd_init)

    a = sub.add_parser("add-cat", help="insert a cat profile")
    a.add_argument("--name", required=True)
    a.add_argument("--dob", required=True, help="YYYY-MM-DD")
    a.add_argument("--weight", required=True, type=float, help="kg")
    a.add_argument("--activity", default="medium", choices=["low", "medium", "high"])
    a.add_argument("--texture-max", default="hard", choices=["soft", "medium", "hard"])
    a.add_argument("--taboo", default="", help="comma-separated ingredient_keys to exclude")
    a.add_argument("--notes", default="")
    a.set_defaults(func=cmd_add_cat)

    sub.add_parser("list-cats", help="list stored cats").set_defaults(func=cmd_list_cats)

    g = sub.add_parser("generate", help="produce top-N recipes for a cat")
    g.add_argument("--cat", required=True)
    g.add_argument("--top", type=int, default=5)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(func=cmd_generate)

    f = sub.add_parser("feedback", help="record how the cat reacted")
    f.add_argument("--cat", required=True)
    f.add_argument("--response", required=True, choices=["ate_all", "half", "refused"])
    f.add_argument("--pick", type=int, default=1,
                   help="index from the most recent `generate` (1-based)")
    f.add_argument("--recipe", default=None,
                   help="JSON dict {ingredient_key: grams}; overrides --pick")
    f.set_defaults(func=cmd_feedback)

    h = sub.add_parser("history", help="recent feedings for a cat")
    h.add_argument("--cat", required=True)
    h.add_argument("--limit", type=int, default=20)
    h.set_defaults(func=cmd_history)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
