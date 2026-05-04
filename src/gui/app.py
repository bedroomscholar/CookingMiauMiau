"""PyWebView desktop UI for the cat-food recipe system.

A warm, elegant single-window app with four screens:

  1. Profiles  — list, add, edit, delete cats; pick taboos.
  2. Generate  — pick a cat, generate top-N candidate recipes,
                 record feedback (which nudges preference weights).
  3. History   — recent feedings per cat.
  4. Library   — browse the merged catalogue and add custom ingredients.

Heavy lifting (DB, recommender, weights, ML model) lives in the
underlying modules. This file is the view+controller layer:
the `Api` class exposes JSON methods to JavaScript via pywebview's
`js_api`, and the embedded HTML/CSS draws the UI.

Run:
    py -m src.gui.app
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import date
from typing import Any

import joblib
import webview

from src.config import INGREDIENTS_CSV, MODELS_DIR
from src.data.library import (
    NUTRIENT_COLUMNS, VALID_CATEGORIES, VALID_TEXTURES,
    add_custom, delete_custom, list_custom_keys, load_ingredients,
)
from src.feedback.db import (
    add_cat, connect, delete_cat, feeding_history, get_cat,
    list_cats, record_feeding, update_cat,
)
from src.feedback.weights import get_weights, update_after_feeding
from src.nutrition.targets import compute_age_years
from src.recommender.filters import applicable_ingredients
from src.recommender.generator import propose
from src.recommender.rank import rank

MODEL_PATH = MODELS_DIR / "gbr.joblib"
ACTIVITIES = ("low", "medium", "high")
TEXTURES = ("soft", "medium", "hard")
RESPONSES = ("ate_all", "half", "refused")


def _parse_dob(raw: str) -> date:
    """Accept the DD-MM-YYYY format the GUI uses; fall back to ISO."""
    s = raw.strip().replace("/", "-").replace(".", "-")
    if not s:
        raise ValueError("date of birth is required")
    parts = s.split("-")
    if len(parts) == 3 and len(parts[0]) == 2 and len(parts[2]) == 4:
        d, m, y = parts
        return date(int(y), int(m), int(d))
    return date.fromisoformat(s)


def _format_recipe_parts(recipe: dict[str, float], ingredients) -> list[dict]:
    parts = []
    for k, g in sorted(recipe.items(), key=lambda kv: -kv[1]):
        display = ingredients.loc[k, "display"] if k in ingredients.index else k
        parts.append({"key": k, "display": display, "grams": float(g)})
    return parts


# --- bridge ------------------------------------------------------------

class Api:
    """Methods on this class are callable from JavaScript as
    `pywebview.api.<method>(...)`. Returned values are JSON-serialised."""

    def __init__(self) -> None:
        # Underscore-prefix every non-method attribute. pywebview walks
        # `dir()` to expose the API to JS, and a public `ingredients`
        # DataFrame causes it to recurse into pandas internals (.T → .T → …).
        self._conn = connect()
        self._ingredients = load_ingredients()
        self._model = joblib.load(MODEL_PATH)
        self._last_ranked: list = []
        self._last_cat_id: int | None = None

    # --- ingredients -------------------------------------------------

    def _reload_ingredients(self) -> None:
        self._ingredients = load_ingredients()

    def list_ingredients(self) -> list[dict]:
        custom = list_custom_keys()
        out = []
        for key, row in self._ingredients.iterrows():
            out.append({
                "key": key,
                "display": str(row.get("display", "")),
                "category": str(row.get("category", "")),
                "texture": str(row.get("texture", "")),
                "kcal": float(row.get("kcal", 0) or 0),
                "protein_g": float(row.get("protein_g", 0) or 0),
                "fat_g": float(row.get("fat_g", 0) or 0),
                "custom": key in custom,
            })
        return out

    def nutrient_columns(self) -> list[str]:
        return list(NUTRIENT_COLUMNS)

    def categories(self) -> list[str]:
        return list(VALID_CATEGORIES)

    def textures(self) -> list[str]:
        return list(VALID_TEXTURES)

    def add_custom_ingredient(self, payload: dict) -> dict:
        try:
            nutrients = {
                k: float(payload.get("nutrients", {}).get(k, 0) or 0)
                for k in NUTRIENT_COLUMNS
            }
            add_custom(
                key=str(payload.get("key", "")).strip(),
                display=str(payload.get("display", "")),
                category=str(payload.get("category", "")),
                texture=str(payload.get("texture", "")),
                nutrients=nutrients,
            )
        except (ValueError, KeyError) as e:
            return {"ok": False, "error": str(e)}
        self._reload_ingredients()
        return {"ok": True}

    def delete_custom_ingredient(self, key: str) -> dict:
        if key not in list_custom_keys():
            return {"ok": False, "error": "Cannot delete a built-in ingredient."}
        try:
            delete_custom(key)
        except KeyError as e:
            return {"ok": False, "error": str(e)}
        self._reload_ingredients()
        return {"ok": True}

    # --- cats --------------------------------------------------------

    def _cat_to_dict(self, c) -> dict:
        return {
            "id": c.id, "name": c.name, "dob": c.dob,
            "weight_kg": c.weight_kg, "activity": c.activity,
            "texture_max": c.texture_max, "notes": c.notes,
            "taboos": list(c.taboos),
        }

    def list_cats(self) -> list[dict]:
        return [self._cat_to_dict(c) for c in list_cats(self._conn)]

    def get_cat(self, name: str) -> dict | None:
        try:
            return self._cat_to_dict(get_cat(self._conn, name))
        except KeyError:
            return None

    def save_cat(self, payload: dict) -> dict:
        try:
            name = str(payload.get("name", "")).strip()
            if not name:
                return {"ok": False, "error": "Name is required."}
            dob = _parse_dob(str(payload.get("dob", "")))
            weight = float(payload.get("weight_kg", 0))
            activity = str(payload.get("activity", "medium"))
            texture_max = str(payload.get("texture_max", "hard"))
            notes = str(payload.get("notes", "") or "")
            taboos = list(payload.get("taboos", []) or [])
            cat_id = payload.get("id")
        except (ValueError, KeyError, TypeError) as e:
            return {"ok": False, "error": f"Invalid input: {e}"}

        try:
            if cat_id in (None, "", 0):
                add_cat(
                    self._conn,
                    name=name, dob=dob, weight_kg=weight,
                    activity=activity, texture_max=texture_max,
                    notes=notes, taboos=taboos,
                )
            else:
                update_cat(
                    self._conn, cat_id=int(cat_id),
                    weight_kg=weight, activity=activity,
                    texture_max=texture_max, notes=notes, taboos=taboos,
                )
        except sqlite3.IntegrityError as e:
            return {"ok": False, "error": str(e)}
        return {"ok": True}

    def delete_cat(self, cat_id: int) -> dict:
        delete_cat(self._conn, int(cat_id))
        return {"ok": True}

    # --- generate / feedback ----------------------------------------

    def generate(self, name: str, top_n: int) -> dict:
        if not name:
            return {"ok": False, "error": "Pick a cat first."}
        try:
            top_n = max(1, int(top_n))
        except (ValueError, TypeError):
            top_n = 5
        try:
            cat = get_cat(self._conn, name)
            age = compute_age_years(date.fromisoformat(cat.dob))
            pool = applicable_ingredients(
                self._ingredients, taboos=cat.taboos, texture_max=cat.texture_max,
            )
            cands = propose(pool, n_candidates=300, seed=None)
            weights = get_weights(self._conn, cat.id)
            ranked = rank(
                cands, model=self._model, age_years=age,
                weights=weights, top_n=top_n, ingredients=self._ingredients,
            )
        except Exception as e:
            return {"ok": False, "error": str(e)}

        self._last_ranked = ranked
        self._last_cat_id = cat.id

        recipes = []
        for i, r in enumerate(ranked, 1):
            recipes.append({
                "index": i,
                "blended": float(r.blended),
                "ml_score": float(r.ml_score),
                "pref_mean": float(r.pref_mean),
                "used_rule_scorer": bool(r.used_rule_scorer),
                "parts": _format_recipe_parts(r.recipe, self._ingredients),
            })
        return {
            "ok": True, "cat": cat.name, "age": float(age), "recipes": recipes,
        }

    def record_feedback(self, pick: int, response: str) -> dict:
        if not self._last_ranked or self._last_cat_id is None:
            return {"ok": False, "error": "Generate recipes first."}
        try:
            pick = int(pick)
        except (ValueError, TypeError):
            return {"ok": False, "error": "Bad pick number."}
        if not (1 <= pick <= len(self._last_ranked)):
            return {"ok": False, "error": f"Pick must be 1..{len(self._last_ranked)}"}
        if response not in RESPONSES:
            return {"ok": False, "error": "Bad response."}
        recipe = self._last_ranked[pick - 1].recipe
        record_feeding(self._conn, cat_id=self._last_cat_id,
                       recipe=recipe, response=response)
        new_w = update_after_feeding(self._conn, cat_id=self._last_cat_id,
                                     recipe=recipe, response=response)
        return {
            "ok": True,
            "weights": {k: float(new_w[k]) for k in recipe},
            "pick": pick, "response": response,
        }

    # --- history ----------------------------------------------------

    def feeding_history(self, name: str, limit: int = 50) -> list[dict]:
        if not name:
            return []
        try:
            cat = get_cat(self._conn, name)
        except KeyError:
            return []
        rows = feeding_history(self._conn, cat.id, limit=int(limit))
        for row in rows:
            row["recipe_parts"] = _format_recipe_parts(row["recipe"], self._ingredients)
        return rows


# --- HTML --------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CookingMiauMiau</title>
<style>
  :root {
    --bg:        #FAF6F0;
    --bg-soft:   #F3ECDF;
    --panel:     #FFFFFF;
    --ink:       #3D2E22;
    --ink-soft:  #6B5847;
    --muted:     #9A8876;
    --line:      #E8DFD3;
    --accent:    #C97B5C;
    --accent-d:  #A85C40;
    --butter:    #E8C9A0;
    --sage:      #7A9471;
    --danger:    #8B1A1F;
    --danger-d:  #6F1418;
    --shadow:    0 4px 24px rgba(89, 60, 38, 0.08);
    --shadow-sm: 0 2px 8px rgba(89, 60, 38, 0.06);
    --radius:    14px;
    --radius-sm: 8px;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0; padding: 0; height: 100%;
    background: var(--bg);
    color: var(--ink);
    font-family: 'Inter', 'Segoe UI', -apple-system, BlinkMacSystemFont, sans-serif;
    font-size: 14px;
    line-height: 1.5;
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
  }
  h1, h2, h3 {
    font-family: 'Playfair Display', 'Georgia', serif;
    font-weight: 600;
    color: var(--ink);
    margin: 0;
    letter-spacing: 0.2px;
  }
  /* Layout */
  .app {
    display: flex; flex-direction: column;
    height: 100vh;
    background:
      radial-gradient(circle at 10% 0%, #F7E8D6 0%, transparent 40%),
      radial-gradient(circle at 100% 100%, #F3DCC3 0%, transparent 35%),
      var(--bg);
  }
  header {
    padding: 20px 32px 0;
    flex-shrink: 0;
  }
  .title-row {
    display: flex; align-items: baseline; gap: 14px;
    margin-bottom: 4px;
  }
  .title-row h1 { font-size: 26px; }
  .title-row .sub {
    color: var(--muted); font-size: 13px;
    font-style: italic;
  }
  .tabs {
    display: flex; gap: 4px;
    margin-top: 18px;
    border-bottom: 1px solid var(--line);
  }
  .tab {
    padding: 10px 22px;
    cursor: pointer;
    color: var(--muted);
    font-weight: 500;
    border: none;
    background: transparent;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.18s, background 0.18s, border-color 0.18s;
    font-family: inherit;
    font-size: 14px;
    letter-spacing: 0.3px;
    border-radius: 6px 6px 0 0;
  }
  /* Hover (inactive): neutral darken, NOT accent — keeps accent reserved
     for the selected tab so the two states never blur together. */
  .tab:hover {
    color: var(--ink);
    background: var(--bg-soft);
  }
  .tab.active {
    color: var(--accent-d);
    font-weight: 600;
    border-bottom: 2.5px solid var(--accent);
    background: transparent;
  }
  .tab.active:hover {
    color: var(--accent-d);
    background: rgba(201, 123, 92, 0.08);
  }
  main {
    flex: 1;
    padding: 24px 32px 28px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .panel {
    background: var(--panel);
    border-radius: var(--radius);
    box-shadow: var(--shadow);
    padding: 24px;
    border: 1px solid var(--line);
  }
  .panel + .panel { margin-top: 16px; }
  .panel h2 {
    font-size: 18px;
    margin-bottom: 14px;
  }
  .panel h3 {
    font-size: 15px;
    color: var(--ink-soft);
    margin-bottom: 8px;
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 11px;
  }
  /* Form controls */
  label { display: block; font-size: 12px; color: var(--muted); margin-bottom: 4px; letter-spacing: 0.3px; }
  input[type=text], input[type=number], input[type=date], select, textarea {
    width: 100%;
    padding: 8px 12px;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--bg);
    color: var(--ink);
    font-family: inherit;
    font-size: 13px;
    transition: border-color 0.15s, background 0.15s;
  }
  input:focus, select:focus, textarea:focus {
    outline: none;
    border-color: var(--accent);
    background: #fff;
  }
  textarea { resize: vertical; min-height: 60px; }
  button {
    padding: 8px 18px;
    border: none;
    border-radius: var(--radius-sm);
    background: var(--accent);
    color: #fff;
    font-family: inherit;
    font-size: 13px;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s, box-shadow 0.15s;
    letter-spacing: 0.3px;
    box-shadow: var(--shadow-sm);
  }
  button:hover { background: var(--accent-d); }
  button:active { transform: translateY(1px); }
  button.ghost {
    background: transparent;
    color: var(--ink-soft);
    border: 1px solid var(--line);
    box-shadow: none;
  }
  button.ghost:hover { background: var(--bg-soft); color: var(--ink); }
  button.danger { background: var(--danger); }
  button.danger:hover { background: var(--danger-d); }
  button.sage { background: var(--sage); }
  button.sage:hover { background: #5F7A57; }
  button:disabled {
    opacity: 0.45;
    cursor: not-allowed;
    box-shadow: none;
  }
  button:disabled:hover { background: var(--danger); }
  button.icon {
    padding: 6px 10px; font-size: 12px;
  }
  .row { display: flex; gap: 12px; align-items: flex-end; flex-wrap: wrap; }
  .row > * { flex: 0 0 auto; }
  .col { display: flex; flex-direction: column; gap: 10px; }
  .grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 12px 18px;
  }
  .grid.three { grid-template-columns: repeat(3, minmax(0,1fr)); }
  .grid.four  { grid-template-columns: repeat(4, minmax(0,1fr)); }
  /* Two-pane layout */
  .split {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 20px;
    flex: 1;
    min-height: 0;
  }
  .scroll {
    overflow-y: auto;
    scrollbar-width: thin;
    scrollbar-color: var(--line) transparent;
  }
  .scroll::-webkit-scrollbar { width: 8px; }
  .scroll::-webkit-scrollbar-thumb { background: var(--line); border-radius: 4px; }
  /* Cat list */
  .cat-list { display: flex; flex-direction: column; gap: 4px; }
  .cat-item {
    padding: 10px 14px;
    border-radius: var(--radius-sm);
    cursor: pointer;
    border: 1px solid transparent;
    transition: background 0.15s, border-color 0.15s;
  }
  .cat-item:hover { background: var(--bg-soft); }
  .cat-item.active {
    background: var(--bg-soft);
    border-color: var(--butter);
    box-shadow: inset 3px 0 0 var(--accent);
  }
  .cat-item .nm { font-weight: 600; color: var(--ink); }
  .cat-item .meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
  /* Taboo chips */
  .chip-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 6px;
    max-height: 240px;
    padding: 6px;
    border: 1px solid var(--line);
    border-radius: var(--radius-sm);
    background: var(--bg);
  }
  .chip {
    padding: 6px 10px;
    border-radius: 999px;
    border: 1px solid var(--line);
    background: var(--panel);
    font-size: 12px;
    cursor: pointer;
    user-select: none;
    transition: all 0.15s;
    color: var(--ink-soft);
    text-align: center;
    line-height: 1.2;
  }
  .chip:hover { border-color: var(--accent); color: var(--accent-d); }
  .chip.on {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
  }
  .chip .ct { font-size: 10px; opacity: 0.7; margin-left: 4px; }
  .chip.custom::before {
    content: '★ ';
    color: var(--butter);
  }
  .chip.on.custom::before { color: #fff; }
  /* Recipe cards */
  .recipe-list { display: flex; flex-direction: column; gap: 12px; }
  .recipe-card {
    padding: 16px 18px;
    border-radius: var(--radius);
    background: var(--bg);
    border: 1px solid var(--line);
    transition: border-color 0.15s, box-shadow 0.15s;
    cursor: pointer;
  }
  .recipe-card:hover {
    border-color: var(--butter);
    box-shadow: var(--shadow-sm);
  }
  .recipe-card.picked {
    border-color: var(--accent);
    background: #fff;
    box-shadow: 0 0 0 2px rgba(201, 123, 92, 0.15);
  }
  .recipe-head {
    display: flex; align-items: center; justify-content: space-between;
    gap: 10px;
    margin-bottom: 10px;
  }
  .recipe-rank {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 22px;
    color: var(--accent-d);
    font-weight: 600;
    width: 36px;
  }
  .recipe-score {
    font-size: 18px;
    font-weight: 600;
    color: var(--ink);
  }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    background: var(--bg-soft);
    color: var(--ink-soft);
  }
  .badge.ml   { background: #E5EFE3; color: #4F6B47; }
  .badge.rule { background: #F5DEC8; color: #8C5A37; }
  .recipe-meta {
    font-size: 11px; color: var(--muted);
    display: flex; gap: 14px; flex-wrap: wrap;
  }
  .recipe-parts { margin-top: 10px; display: flex; flex-wrap: wrap; gap: 6px; }
  .part {
    padding: 4px 10px;
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 999px;
    font-size: 12px;
    color: var(--ink-soft);
  }
  .part .g { color: var(--accent-d); font-weight: 600; margin-left: 4px; }
  /* Tables */
  .table {
    width: 100%; border-collapse: collapse;
    font-size: 13px;
  }
  .table th {
    text-align: left;
    padding: 8px 10px;
    color: var(--muted);
    font-weight: 600;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    border-bottom: 1px solid var(--line);
    background: var(--bg-soft);
  }
  .table td {
    padding: 10px;
    border-bottom: 1px solid var(--line);
    color: var(--ink-soft);
  }
  .table tr:hover td { background: var(--bg-soft); cursor: pointer; }
  .table tr.selected td { background: #FBEFE0; }
  .resp {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
  }
  .resp.ate_all { background: #E5EFE3; color: #4F6B47; }
  .resp.half    { background: #FBEFE0; color: #8C5A37; }
  .resp.refused { background: #F4DAD9; color: #8B3A37; }
  /* Empty / status */
  .empty {
    padding: 36px 18px;
    text-align: center;
    color: var(--muted);
    font-style: italic;
  }
  .status {
    padding: 8px 14px;
    margin: 10px 0;
    border-radius: var(--radius-sm);
    font-size: 13px;
    display: none;
  }
  .status.show { display: block; }
  .status.ok    { background: #E5EFE3; color: #4F6B47; }
  .status.err   { background: #F4DAD9; color: #8B3A37; }
  /* Toast */
  .toast {
    position: fixed; right: 24px; bottom: 24px;
    background: var(--ink); color: #fff;
    padding: 12px 18px;
    border-radius: var(--radius-sm);
    box-shadow: var(--shadow);
    font-size: 13px;
    opacity: 0;
    transform: translateY(8px);
    transition: opacity 0.25s, transform 0.25s;
    pointer-events: none;
    max-width: 360px;
    z-index: 999;
  }
  .toast.show { opacity: 1; transform: translateY(0); }
  .toast.err { background: var(--danger); }
  /* Tab pages */
  .page { display: none; flex: 1; min-height: 0; flex-direction: column; }
  .page.active { display: flex; }
  /* Generate layout */
  .gen-bar { display: flex; gap: 14px; align-items: flex-end; }
  .gen-bar .grow { flex: 1; }
  .feedback-bar {
    margin-top: 14px;
    padding: 14px 16px;
    background: var(--bg-soft);
    border-radius: var(--radius);
    display: flex; gap: 10px; align-items: center;
    border: 1px dashed var(--butter);
  }
  .feedback-bar .lbl {
    font-size: 12px; color: var(--ink-soft);
    margin-right: 6px;
  }
  /* History/Library tabs use full panel, generate uses split */
  .full-panel { flex: 1; min-height: 0; display: flex; flex-direction: column; }
  .full-panel .scroll { flex: 1; }
  /* Library forms */
  .library-grid {
    display: grid;
    grid-template-columns: 1.2fr 1fr;
    gap: 16px;
    flex: 1;
    min-height: 0;
  }
</style>
</head>
<body>
<div class="app">
  <header>
    <div class="title-row">
      <h1>CookingMiauMiau</h1>
      <span class="sub">thank you my dear human</span>
    </div>
    <nav class="tabs">
      <button class="tab active" data-page="profiles">Profiles</button>
      <button class="tab" data-page="generate">Generate</button>
      <button class="tab" data-page="history">History</button>
      <button class="tab" data-page="library">Library</button>
    </nav>
  </header>
  <main>
    <!-- PROFILES -->
    <section class="page active" id="page-profiles">
      <div class="split">
        <div class="panel" style="display:flex; flex-direction:column; min-height:0;">
          <h2>Cats</h2>
          <div class="cat-list scroll" id="catList" style="flex:1; min-height:0;"></div>
          <div style="display:flex; gap:8px; margin-top:12px;">
            <button id="catNew" class="ghost">+ New</button>
            <button id="catDelete" class="danger">Delete</button>
          </div>
        </div>
        <div class="panel scroll" style="min-height:0;">
          <h2 id="profileTitle">New profile</h2>
          <form id="profileForm" autocomplete="off">
            <div class="grid">
              <div><label>Name</label><input type="text" id="pf-name"></div>
              <div><label>Date of Birth (DD-MM-YYYY)</label><input type="text" id="pf-dob" placeholder="DD-MM-YYYY" maxlength="10"></div>
              <div><label>Weight (kg)</label><input type="number" step="0.1" id="pf-weight"></div>
              <div><label>Activity</label><select id="pf-activity"></select></div>
              <div><label>Texture max</label><select id="pf-texture"></select></div>
              <div><label>Notes</label><input type="text" id="pf-notes"></div>
            </div>
            <h3 style="margin-top:18px;">Taboos · click to toggle</h3>
            <div class="chip-grid scroll" id="tabooChips"></div>
            <div style="display:flex; justify-content:flex-end; margin-top:14px;">
              <button type="submit" id="profileSave">Save profile</button>
            </div>
          </form>
        </div>
      </div>
    </section>

    <!-- GENERATE -->
    <section class="page" id="page-generate">
      <div class="panel" style="margin-bottom:16px;">
        <div class="gen-bar">
          <div class="grow" style="max-width:280px;">
            <label>Cat</label>
            <select id="gn-cat"></select>
          </div>
          <div style="width:90px;">
            <label>Top N</label>
            <input type="number" id="gn-top" value="5" min="1" max="10">
          </div>
          <button id="gn-go">Generate recipes</button>
        </div>
        <div class="feedback-bar" id="feedbackBar" style="display:none;">
          <span class="lbl">After serving recipe</span>
          <select id="fb-pick" style="width:70px;"></select>
          <span class="lbl" style="margin-left:6px;">the cat</span>
          <button class="sage" data-resp="ate_all">ate it all</button>
          <button class="ghost" data-resp="half">ate half</button>
          <button class="danger" data-resp="refused">refused</button>
        </div>
      </div>
      <div class="panel full-panel">
        <h3 id="gn-summary" style="margin-bottom:8px;">Pick a cat and generate to begin</h3>
        <div class="scroll" style="flex:1;">
          <div class="recipe-list" id="recipeList"></div>
        </div>
      </div>
    </section>

    <!-- HISTORY -->
    <section class="page" id="page-history">
      <div class="panel" style="margin-bottom:16px;">
        <div class="gen-bar">
          <div class="grow" style="max-width:280px;">
            <label>Cat</label>
            <select id="hi-cat"></select>
          </div>
          <button id="hi-refresh" class="ghost">Refresh</button>
        </div>
      </div>
      <div class="panel full-panel">
        <div class="scroll" style="flex:1;">
          <table class="table" id="historyTable">
            <thead><tr>
              <th style="width:50px;">#</th>
              <th style="width:160px;">Served at</th>
              <th style="width:110px;">Response</th>
              <th>Recipe</th>
            </tr></thead>
            <tbody></tbody>
          </table>
          <div class="empty" id="historyEmpty" style="display:none;">No feedings recorded yet for this cat.</div>
        </div>
      </div>
    </section>

    <!-- LIBRARY -->
    <section class="page" id="page-library">
      <div class="library-grid">
        <div class="panel scroll" style="min-height:0;">
          <h2>Catalogue</h2>
          <p style="color:var(--muted); font-size:12px; margin:0 0 8px;">★ marks user-added items.</p>
          <table class="table" id="ingredientTable">
            <thead><tr>
              <th>Key</th><th>Display</th><th>Cat.</th><th>Tex.</th>
              <th style="text-align:right;">kcal</th>
              <th style="text-align:right;">protein</th>
              <th style="text-align:right;">fat</th>
            </tr></thead>
            <tbody></tbody>
          </table>
          <div style="margin-top:10px;">
            <button id="lib-delete" class="danger" disabled>Delete selected (custom only)</button>
          </div>
        </div>
        <div class="panel scroll" style="min-height:0;">
          <h2>Add a custom ingredient</h2>
          <form id="customForm" autocomplete="off">
            <div class="grid">
              <div><label>Key (lowercase)</label><input type="text" id="cf-key"></div>
              <div><label>Display name</label><input type="text" id="cf-display"></div>
              <div><label>Category</label><select id="cf-category"></select></div>
              <div><label>Texture</label><select id="cf-texture"></select></div>
            </div>
            <h3 style="margin-top:14px;">Per 100 g as-fed nutrients</h3>
            <div class="grid three" id="nutrientGrid"></div>
            <div style="display:flex; justify-content:flex-end; margin-top:14px;">
              <button type="submit">Add to library</button>
            </div>
          </form>
        </div>
      </div>
    </section>
  </main>
</div>
<div class="toast" id="toast"></div>

<script>
const api = () => window.pywebview.api;

let state = {
  cats: [],
  ingredients: [],
  selectedCatId: null,
  editingTaboos: new Set(),
  selectedIngredient: null,
  lastRecipes: [],
};

function $(sel, root=document) { return root.querySelector(sel); }
function $$(sel, root=document) { return Array.from(root.querySelectorAll(sel)); }

function toast(msg, kind='') {
  const el = $('#toast');
  el.textContent = msg;
  el.className = 'toast show ' + kind;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.classList.remove('show'); }, 2800);
}

// Tab switching
$$('.tab').forEach(btn => {
  btn.addEventListener('click', () => {
    $$('.tab').forEach(t => t.classList.remove('active'));
    $$('.page').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    $('#page-' + btn.dataset.page).classList.add('active');
    if (btn.dataset.page === 'history') refreshHistory();
  });
});

// --- Profiles ---------------------------------------------------------

function renderCatList() {
  const list = $('#catList');
  list.innerHTML = '';
  if (!state.cats.length) {
    const e = document.createElement('div');
    e.className = 'empty';
    e.textContent = 'No cats yet.\nClick "+ New" to add one.';
    e.style.whiteSpace = 'pre-line';
    list.appendChild(e);
    return;
  }
  state.cats.forEach(c => {
    const div = document.createElement('div');
    div.className = 'cat-item' + (c.id === state.selectedCatId ? ' active' : '');
    div.innerHTML = `<div class="nm">${escapeHtml(c.name)}</div>
      <div class="meta">${c.weight_kg.toFixed(1)} kg · ${escapeHtml(c.activity)} · texture ≤ ${escapeHtml(c.texture_max)}</div>`;
    div.addEventListener('click', () => selectCat(c.id));
    list.appendChild(div);
  });
}

function renderTabooChips() {
  const grid = $('#tabooChips');
  grid.innerHTML = '';
  state.ingredients.forEach(ing => {
    const chip = document.createElement('div');
    let cls = 'chip';
    if (state.editingTaboos.has(ing.key)) cls += ' on';
    if (ing.custom) cls += ' custom';
    chip.className = cls;
    chip.innerHTML = `${escapeHtml(ing.display)}<span class="ct">${escapeHtml(ing.category)}</span>`;
    chip.addEventListener('click', () => {
      if (state.editingTaboos.has(ing.key)) state.editingTaboos.delete(ing.key);
      else state.editingTaboos.add(ing.key);
      renderTabooChips();
    });
    grid.appendChild(chip);
  });
}

function fillSelect(sel, items, selected) {
  sel.innerHTML = '';
  items.forEach(v => {
    const o = document.createElement('option');
    o.value = v; o.textContent = v;
    if (v === selected) o.selected = true;
    sel.appendChild(o);
  });
}

function clearProfileForm() {
  state.selectedCatId = null;
  state.editingTaboos = new Set();
  $('#profileTitle').textContent = 'New profile';
  $('#pf-name').value = '';
  $('#pf-name').readOnly = false;
  $('#pf-dob').value = '';
  $('#pf-weight').value = '';
  $('#pf-activity').value = 'medium';
  $('#pf-texture').value = 'hard';
  $('#pf-notes').value = '';
  renderCatList();
  renderTabooChips();
}

function selectCat(id) {
  const cat = state.cats.find(c => c.id === id);
  if (!cat) return;
  state.selectedCatId = id;
  state.editingTaboos = new Set(cat.taboos);
  $('#profileTitle').textContent = 'Edit · ' + cat.name;
  $('#pf-name').value = cat.name;
  $('#pf-name').readOnly = true; // name is unique key — can't rename here
  $('#pf-dob').value = isoToDmy(cat.dob);
  $('#pf-weight').value = cat.weight_kg;
  $('#pf-activity').value = cat.activity;
  $('#pf-texture').value = cat.texture_max;
  $('#pf-notes').value = cat.notes || '';
  renderCatList();
  renderTabooChips();
}

$('#catNew').addEventListener('click', clearProfileForm);

$('#catDelete').addEventListener('click', async () => {
  if (state.selectedCatId == null) return;
  const cat = state.cats.find(c => c.id === state.selectedCatId);
  if (!cat) return;
  if (!confirm(`Delete ${cat.name} and all their feedings?`)) return;
  await api().delete_cat(state.selectedCatId);
  await reloadCats();
  clearProfileForm();
  toast(`Deleted ${cat.name}.`);
});

$('#profileForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const payload = {
    id: state.selectedCatId,
    name: $('#pf-name').value.trim(),
    dob: $('#pf-dob').value,
    weight_kg: parseFloat($('#pf-weight').value),
    activity: $('#pf-activity').value,
    texture_max: $('#pf-texture').value,
    notes: $('#pf-notes').value,
    taboos: Array.from(state.editingTaboos),
  };
  const res = await api().save_cat(payload);
  if (!res.ok) { toast(res.error, 'err'); return; }
  toast(`Saved ${payload.name}.`);
  await reloadCats();
  // Re-select by name
  const c = state.cats.find(x => x.name === payload.name);
  if (c) selectCat(c.id);
});

// --- Generate ---------------------------------------------------------

$('#gn-go').addEventListener('click', async () => {
  const name = $('#gn-cat').value;
  const top = parseInt($('#gn-top').value || '5', 10);
  if (!name) { toast('Add a cat in Profiles first.', 'err'); return; }
  $('#gn-summary').textContent = 'Generating…';
  $('#recipeList').innerHTML = '';
  const res = await api().generate(name, top);
  if (!res.ok) { toast(res.error, 'err'); $('#gn-summary').textContent = 'Generation failed.'; return; }
  state.lastRecipes = res.recipes;
  $('#gn-summary').textContent = `Top ${res.recipes.length} for ${res.cat} · age ${res.age.toFixed(1)} y`;
  renderRecipes(res.recipes);
  // populate feedback selector
  const fb = $('#fb-pick');
  fb.innerHTML = '';
  res.recipes.forEach(r => {
    const o = document.createElement('option');
    o.value = r.index; o.textContent = '#' + r.index;
    fb.appendChild(o);
  });
  $('#feedbackBar').style.display = res.recipes.length ? 'flex' : 'none';
});

function renderRecipes(recipes) {
  const list = $('#recipeList');
  list.innerHTML = '';
  if (!recipes.length) {
    list.innerHTML = '<div class="empty">No recipes returned.</div>';
    return;
  }
  recipes.forEach(r => {
    const card = document.createElement('div');
    card.className = 'recipe-card';
    card.dataset.index = r.index;
    const tag = r.used_rule_scorer
      ? '<span class="badge rule">rule scorer</span>'
      : '<span class="badge ml">ML model</span>';
    const parts = r.parts.map(p =>
      `<span class="part">${escapeHtml(p.display)}<span class="g">${p.grams.toFixed(1)} g</span></span>`
    ).join('');
    card.innerHTML = `
      <div class="recipe-head">
        <div style="display:flex; align-items:baseline; gap:14px;">
          <span class="recipe-rank">${r.index}</span>
          <span class="recipe-score">${r.blended.toFixed(1)}</span>
          ${tag}
        </div>
        <div class="recipe-meta">
          <span>nutrition ${r.ml_score.toFixed(1)}</span>
          <span>preference ×${r.pref_mean.toFixed(2)}</span>
        </div>
      </div>
      <div class="recipe-parts">${parts}</div>
    `;
    card.addEventListener('click', () => {
      $$('.recipe-card').forEach(c => c.classList.remove('picked'));
      card.classList.add('picked');
      $('#fb-pick').value = r.index;
    });
    list.appendChild(card);
  });
}

$$('[data-resp]').forEach(btn => {
  btn.addEventListener('click', async () => {
    const pick = parseInt($('#fb-pick').value, 10);
    const resp = btn.dataset.resp;
    const res = await api().record_feedback(pick, resp);
    if (!res.ok) { toast(res.error, 'err'); return; }
    const changes = Object.entries(res.weights)
      .map(([k, v]) => `${k}:${v.toFixed(2)}`).join(', ');
    toast(`Logged "${resp}" · ${changes}`);
  });
});

// --- History ----------------------------------------------------------

$('#hi-refresh').addEventListener('click', refreshHistory);
$('#hi-cat').addEventListener('change', refreshHistory);

async function refreshHistory() {
  const name = $('#hi-cat').value;
  const tbody = $('#historyTable tbody');
  tbody.innerHTML = '';
  if (!name) {
    $('#historyEmpty').style.display = 'block';
    $('#historyEmpty').textContent = 'No cats yet.';
    return;
  }
  const rows = await api().feeding_history(name, 50);
  if (!rows.length) {
    $('#historyEmpty').style.display = 'block';
    $('#historyEmpty').textContent = 'No feedings yet for this cat.';
    return;
  }
  $('#historyEmpty').style.display = 'none';
  rows.forEach(row => {
    const tr = document.createElement('tr');
    const parts = row.recipe_parts.map(p =>
      `<span class="part">${escapeHtml(p.display)}<span class="g">${p.grams.toFixed(1)} g</span></span>`
    ).join(' ');
    tr.innerHTML = `
      <td>${row.id}</td>
      <td>${escapeHtml(row.served_at.replace('T', ' '))}</td>
      <td><span class="resp ${row.response}">${escapeHtml(row.response)}</span></td>
      <td>${parts}</td>`;
    tbody.appendChild(tr);
  });
}

// --- Library ----------------------------------------------------------

function renderIngredientTable() {
  const tbody = $('#ingredientTable tbody');
  tbody.innerHTML = '';
  state.ingredients.forEach(ing => {
    const tr = document.createElement('tr');
    if (state.selectedIngredient === ing.key) tr.classList.add('selected');
    tr.innerHTML = `
      <td>${ing.custom ? '★ ' : ''}${escapeHtml(ing.key)}</td>
      <td>${escapeHtml(ing.display)}</td>
      <td>${escapeHtml(ing.category)}</td>
      <td>${escapeHtml(ing.texture)}</td>
      <td style="text-align:right;">${ing.kcal.toFixed(0)}</td>
      <td style="text-align:right;">${ing.protein_g.toFixed(1)}</td>
      <td style="text-align:right;">${ing.fat_g.toFixed(1)}</td>`;
    tr.addEventListener('click', () => {
      state.selectedIngredient = ing.key;
      $('#lib-delete').disabled = !ing.custom;
      renderIngredientTable();
    });
    tbody.appendChild(tr);
  });
}

$('#lib-delete').addEventListener('click', async () => {
  const k = state.selectedIngredient;
  if (!k) return;
  if (!confirm(`Remove custom ingredient ${k}?`)) return;
  const res = await api().delete_custom_ingredient(k);
  if (!res.ok) { toast(res.error, 'err'); return; }
  state.selectedIngredient = null;
  $('#lib-delete').disabled = true;
  await reloadIngredients();
  toast(`Removed ${k}.`);
});

$('#customForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const nutrients = {};
  $$('#nutrientGrid input').forEach(inp => {
    nutrients[inp.dataset.col] = parseFloat(inp.value || '0');
  });
  const payload = {
    key: $('#cf-key').value.trim(),
    display: $('#cf-display').value,
    category: $('#cf-category').value,
    texture: $('#cf-texture').value,
    nutrients,
  };
  const res = await api().add_custom_ingredient(payload);
  if (!res.ok) { toast(res.error, 'err'); return; }
  toast(`Added ${payload.key}.`);
  $('#cf-key').value = '';
  $('#cf-display').value = '';
  $$('#nutrientGrid input').forEach(inp => inp.value = '0');
  await reloadIngredients();
});

// --- Reload helpers ---------------------------------------------------

async function reloadCats() {
  state.cats = await api().list_cats();
  renderCatList();
  // Refresh dropdowns
  const names = state.cats.map(c => c.name);
  fillSelect($('#gn-cat'), names.length ? names : [], names[0]);
  fillSelect($('#hi-cat'), names.length ? names : [], names[0]);
}

async function reloadIngredients() {
  state.ingredients = await api().list_ingredients();
  renderIngredientTable();
  if (state.selectedCatId == null) renderTabooChips();
  else {
    // keep editingTaboos as-is, just re-render
    renderTabooChips();
  }
}

// --- Boot -------------------------------------------------------------

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, ch => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
  }[ch]));
}

// Backend stores DOB as ISO YYYY-MM-DD; the form shows DD-MM-YYYY.
function isoToDmy(iso) {
  if (!iso) return '';
  const m = String(iso).match(/^(\d{4})-(\d{2})-(\d{2})$/);
  return m ? `${m[3]}-${m[2]}-${m[1]}` : String(iso);
}

async function boot() {
  // Selects
  fillSelect($('#pf-activity'), ['low', 'medium', 'high'], 'medium');
  fillSelect($('#pf-texture'),  ['soft', 'medium', 'hard'], 'hard');

  const cats = await api().categories();
  const texs = await api().textures();
  const cols = await api().nutrient_columns();
  fillSelect($('#cf-category'), cats, 'protein');
  fillSelect($('#cf-texture'),  texs, 'soft');
  const grid = $('#nutrientGrid');
  cols.forEach(col => {
    const wrap = document.createElement('div');
    wrap.innerHTML = `<label>${escapeHtml(col)}</label>
      <input type="number" step="0.1" data-col="${escapeHtml(col)}" value="0">`;
    grid.appendChild(wrap);
  });

  await reloadIngredients();
  await reloadCats();
  clearProfileForm();
}

// pywebview injects api asynchronously. Guard so boot only fires once
// (both 'pywebviewready' and our timeout fallback can race).
let booted = false;
function safeBoot() {
  if (booted) return;
  if (!(window.pywebview && window.pywebview.api)) return;
  booted = true;
  boot().catch(err => toast('Boot failed: ' + err.message, 'err'));
}
window.addEventListener('pywebviewready', safeBoot);
setTimeout(safeBoot, 400);
</script>
</body>
</html>
"""


# --- entry point -------------------------------------------------------

def _missing_data_message() -> str | None:
    if not INGREDIENTS_CSV.exists():
        return (f"{INGREDIENTS_CSV} not found.\n"
                "Run `py -m src.data.build_dataset` first.")
    if not MODEL_PATH.exists():
        return (f"{MODEL_PATH} not found.\n"
                "Run `py -m src.ml.train_alt` first.")
    return None


def main() -> None:
    msg = _missing_data_message()
    if msg:
        print(msg, file=sys.stderr)
        sys.exit(1)

    api = Api()
    webview.create_window(
        "CookingMiauMiau",
        html=INDEX_HTML,
        js_api=api,
        width=1180,
        height=780,
        min_size=(960, 640),
        background_color="#FAF6F0",
    )
    webview.start()


if __name__ == "__main__":
    main()
