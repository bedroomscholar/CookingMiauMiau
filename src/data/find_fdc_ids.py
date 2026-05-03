"""Interactive helper to pick USDA fdcIds for the shortlist.

Run:
    py -m src.data.find_fdc_ids

For each shortlist entry still at fdc_id=0 (and not a pure-supplement
manual override), queries USDA /foods/search, prints the top candidates,
and lets you pick one. Picks are checkpointed to data/raw/usda/_picks.json
so the run is resumable. When the loop finishes (or you quit early), the
helper rewrites src/data/ingredient_shortlist.py in place, injecting
fdc_id=NNN into the matching IngredientSpec lines.

Commands at each prompt:
    <n>           pick result number n
    r <query>     re-search with a new query string
    c <id>        enter a custom fdcId directly (e.g. one you found on the website)
    s             skip this ingredient (leave fdc_id=0 for now)
    q             quit and save what you have so far
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import requests

from src.config import RAW_DIR, ROOT, usda_api_key
from src.data.build_dataset import MANUAL_OVERRIDES
from src.data.ingredient_shortlist import SHORTLIST, IngredientSpec

SEARCH_URL = "https://api.nal.usda.gov/fdc/v1/foods/search"
SHORTLIST_PY = ROOT / "src" / "data" / "ingredient_shortlist.py"
CHECKPOINT = RAW_DIR / "_picks.json"

# Pure supplements with a full manual nutrient override don't need an fdcId.
FULL_OVERRIDE_KEYS = {
    "taurine_powder", "calcium_carb", "eggshell_powder",
    "kelp_powder", "brewers_yeast",
}

# Default query hints — display name plus a state qualifier so the
# top SR Legacy / Foundation hit tends to be the raw / cooked form we want.
QUERY_HINTS: dict[str, str] = {
    "chicken_breast":  "chicken breast raw",
    "chicken_thigh":   "chicken thigh raw",
    "turkey_breast":   "turkey breast raw",
    "turkey_thigh":    "turkey thigh raw",
    "beef_lean":       "beef ground lean raw",
    "pork_lean":       "pork loin lean raw",
    "rabbit":          "rabbit domesticated meat raw",
    "duck":            "duck meat only raw",
    "salmon":          "salmon atlantic raw",
    "sardine":         "sardine canned oil drained",
    "tuna_light":      "tuna light canned water drained",
    "cod":             "cod atlantic raw",
    "chicken_liver":   "chicken liver raw",
    "chicken_heart":   "chicken heart raw",
    "beef_liver":      "beef liver raw",
    "beef_kidney":     "beef kidney raw",
    "egg_whole":       "egg whole cooked hard-boiled",
    "egg_yolk":        "egg yolk raw",
    "white_rice":      "rice white cooked",
    "oats":            "oats cooked",
    "pumpkin":         "pumpkin cooked boiled drained",
    "sweet_potato":    "sweet potato cooked baked",
    "carrot":          "carrot cooked boiled drained",
    "zucchini":        "zucchini cooked boiled drained",
    "salmon_oil":      "fish oil salmon",
    "olive_oil":       "oil olive salad or cooking",
}


def search(query: str, api_key: str, page_size: int = 8) -> list[dict]:
    """Hit USDA search, prefer SR Legacy + Foundation (cleanest raw-ingredient records)."""
    r = requests.get(
        SEARCH_URL,
        params={
            "api_key": api_key,
            "query": query,
            "dataType": ["Foundation", "SR Legacy"],
            "pageSize": page_size,
        },
        timeout=20,
    )
    r.raise_for_status()
    foods = r.json().get("foods", [])
    if not foods:
        # Fall back to broader datasets if the curated ones returned nothing.
        r = requests.get(
            SEARCH_URL,
            params={"api_key": api_key, "query": query, "pageSize": page_size},
            timeout=20,
        )
        r.raise_for_status()
        foods = r.json().get("foods", [])
    return foods


def print_results(foods: list[dict]) -> None:
    if not foods:
        print("  (no results)")
        return
    for i, f in enumerate(foods, 1):
        desc = (f.get("description") or "").strip()
        dtype = f.get("dataType", "?")
        fid = f.get("fdcId")
        brand = f.get("brandOwner") or ""
        extra = f"  [{brand}]" if brand else ""
        print(f"  [{i}] {fid:>7}  {dtype:<12}  {desc}{extra}")


def load_checkpoint() -> dict[str, int]:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    return {}


def save_checkpoint(picks: dict[str, int]) -> None:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    CHECKPOINT.write_text(json.dumps(picks, indent=2), encoding="utf-8")


def prompt_loop(spec: IngredientSpec, api_key: str) -> int | None:
    """Return chosen fdcId, or None to skip. Raises SystemExit on 'q'."""
    query = QUERY_HINTS.get(spec.key, spec.display.lower())
    print(f"\n=== {spec.key}  ({spec.display}, {spec.category}/{spec.texture})")
    foods: list[dict] = []
    while True:
        if not foods:
            print(f"  query: {query!r}")
            try:
                foods = search(query, api_key)
            except requests.RequestException as e:
                print(f"  search failed: {e}")
                foods = []
            print_results(foods)

        try:
            cmd = input("  pick> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\nAborted by user.")

        if not cmd:
            continue
        if cmd == "q":
            raise _QuitAndSave()
        if cmd == "s":
            return None
        if cmd.startswith("r "):
            query = cmd[2:].strip() or query
            foods = []
            continue
        if cmd.startswith("c "):
            try:
                return int(cmd[2:].strip())
            except ValueError:
                print("  not a valid integer fdcId")
                continue
        if cmd.isdigit():
            n = int(cmd)
            if 1 <= n <= len(foods):
                return int(foods[n - 1]["fdcId"])
            print(f"  out of range (1..{len(foods)})")
            continue
        print("  commands: <n> | r <query> | c <id> | s | q")


class _QuitAndSave(Exception):
    pass


def rewrite_shortlist(picks: dict[str, int]) -> int:
    """Inject `, fdc_id=NNN` into each matching IngredientSpec line. Idempotent.

    Done per-line by string surgery rather than regex over the whole file —
    display names can contain parens like "Duck (no skin)" which trip up
    naive `[^)]*` regexes.
    """
    lines = SHORTLIST_PY.read_text(encoding="utf-8").splitlines(keepends=True)
    remaining = {k: v for k, v in picks.items() if v > 0}
    n = 0
    for i, line in enumerate(lines):
        for key, fdc_id in list(remaining.items()):
            needle = f'IngredientSpec("{key}",'
            if needle not in line:
                continue
            if "fdc_id" in line:
                del remaining[key]
                break
            # Inject before the IngredientSpec call's closing paren — the last
            # ')' on the line, since the trailing ',' (if any) follows it.
            close_idx = line.rfind(")")
            if close_idx == -1:
                break
            lines[i] = line[:close_idx] + f", fdc_id={fdc_id}" + line[close_idx:]
            del remaining[key]
            n += 1
            break
    for key in remaining:
        print(f"  WARN: could not locate IngredientSpec line for {key}", file=sys.stderr)
    SHORTLIST_PY.write_text("".join(lines), encoding="utf-8")
    return n


def main() -> None:
    api_key = usda_api_key()  # fails loud if missing
    picks = load_checkpoint()

    pending = [
        s for s in SHORTLIST
        if s.fdc_id == 0
        and s.key not in FULL_OVERRIDE_KEYS
        and s.key not in picks
    ]
    if not pending:
        print("Nothing to pick — all shortlist items already have fdcIds or full overrides.")
        if picks:
            print(f"Applying {len(picks)} cached pick(s) to ingredient_shortlist.py …")
            n = rewrite_shortlist(picks)
            print(f"Rewrote {n} line(s).")
        return

    print(f"Need fdcIds for {len(pending)} ingredient(s). "
          f"Checkpoint: {CHECKPOINT.relative_to(ROOT)}")
    quit_early = False
    try:
        for spec in pending:
            chosen = prompt_loop(spec, api_key)
            if chosen is not None:
                picks[spec.key] = chosen
                save_checkpoint(picks)
                print(f"  -> saved {spec.key} = {chosen}")
            else:
                print(f"  -> skipped {spec.key}")
    except _QuitAndSave:
        quit_early = True
        print("\nQuitting early — applying picks collected so far …")

    n = rewrite_shortlist(picks)
    print(f"\nRewrote {n} IngredientSpec line(s) in {SHORTLIST_PY.relative_to(ROOT)}.")
    if quit_early:
        print("Re-run `py -m src.data.find_fdc_ids` to resume the rest.")
    else:
        print("All picks applied. Next: `py -m src.data.build_dataset`.")


if __name__ == "__main__":
    main()
