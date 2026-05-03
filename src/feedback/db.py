"""SQLite schema and small CRUD helpers for cat profiles, taboos,
feedings, and per-ingredient preference weights.

Design:
  - Path defaults to src.config.DB_PATH but can be overridden (tests pass
    a tmp path).
  - init_db() is idempotent (CREATE TABLE IF NOT EXISTS) and called lazily.
  - Ingredients are NOT mirrored here — the CSV at INGREDIENTS_CSV is the
    source of truth. We only store the ingredient_key string as a soft FK.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS cats (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    name         TEXT    UNIQUE NOT NULL,
    dob          TEXT    NOT NULL,                     -- ISO YYYY-MM-DD
    weight_kg    REAL    NOT NULL,
    activity     TEXT    NOT NULL DEFAULT 'medium',    -- low/medium/high
    texture_max  TEXT    NOT NULL DEFAULT 'hard',      -- soft|medium|hard
    notes        TEXT    NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS cat_taboos (
    cat_id          INTEGER NOT NULL,
    ingredient_key  TEXT    NOT NULL,
    PRIMARY KEY (cat_id, ingredient_key),
    FOREIGN KEY (cat_id) REFERENCES cats(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedings (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    cat_id       INTEGER NOT NULL,
    served_at    TEXT    NOT NULL,                     -- ISO timestamp
    recipe_json  TEXT    NOT NULL,
    response     TEXT    NOT NULL,                     -- ate_all|half|refused
    FOREIGN KEY (cat_id) REFERENCES cats(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS preference_weights (
    cat_id          INTEGER NOT NULL,
    ingredient_key  TEXT    NOT NULL,
    weight          REAL    NOT NULL DEFAULT 1.0,
    PRIMARY KEY (cat_id, ingredient_key),
    FOREIGN KEY (cat_id) REFERENCES cats(id) ON DELETE CASCADE
);
"""

VALID_RESPONSES = ("ate_all", "half", "refused")
VALID_TEXTURES = ("soft", "medium", "hard")


@dataclass(frozen=True)
class Cat:
    id: int
    name: str
    dob: str             # ISO YYYY-MM-DD
    weight_kg: float
    activity: str
    texture_max: str
    notes: str
    taboos: tuple[str, ...] = ()


def connect(path: Path | str | None = None) -> sqlite3.Connection:
    """Open (and lazily init) a SQLite connection.

    `check_same_thread=False` lets the same connection be reused from
    pywebview's worker threads (each JS->Python call runs on a thread
    from the bridge pool). The app is single-user so concurrent writes
    are rare; SQLite's own locking keeps them safe.
    """
    target = Path(path) if path else DB_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    return conn


def add_cat(
    conn: sqlite3.Connection,
    *,
    name: str,
    dob: str | date,
    weight_kg: float,
    activity: str = "medium",
    texture_max: str = "hard",
    notes: str = "",
    taboos: list[str] | None = None,
) -> int:
    if texture_max not in VALID_TEXTURES:
        raise ValueError(f"texture_max must be one of {VALID_TEXTURES}")
    dob_str = dob.isoformat() if isinstance(dob, date) else str(dob)
    cur = conn.execute(
        "INSERT INTO cats(name, dob, weight_kg, activity, texture_max, notes) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (name, dob_str, float(weight_kg), activity, texture_max, notes),
    )
    cat_id = int(cur.lastrowid)
    for k in taboos or []:
        conn.execute(
            "INSERT OR IGNORE INTO cat_taboos(cat_id, ingredient_key) VALUES (?, ?)",
            (cat_id, k),
        )
    conn.commit()
    return cat_id


def update_cat(
    conn: sqlite3.Connection,
    *,
    cat_id: int,
    weight_kg: float | None = None,
    activity: str | None = None,
    texture_max: str | None = None,
    notes: str | None = None,
    taboos: list[str] | None = None,
) -> None:
    """Patch a cat in place. None means 'leave unchanged'. taboos=[...] replaces."""
    fields = {
        "weight_kg": weight_kg, "activity": activity,
        "texture_max": texture_max, "notes": notes,
    }
    sets = {k: v for k, v in fields.items() if v is not None}
    if texture_max is not None and texture_max not in VALID_TEXTURES:
        raise ValueError(f"texture_max must be one of {VALID_TEXTURES}")
    if sets:
        cols = ", ".join(f"{k} = ?" for k in sets)
        conn.execute(f"UPDATE cats SET {cols} WHERE id = ?",
                     (*sets.values(), cat_id))
    if taboos is not None:
        conn.execute("DELETE FROM cat_taboos WHERE cat_id = ?", (cat_id,))
        for k in taboos:
            conn.execute(
                "INSERT OR IGNORE INTO cat_taboos(cat_id, ingredient_key) VALUES (?, ?)",
                (cat_id, k),
            )
    conn.commit()


def delete_cat(conn: sqlite3.Connection, cat_id: int) -> None:
    """Cascade-delete a cat and all their taboos/feedings/weights."""
    conn.execute("DELETE FROM cats WHERE id = ?", (cat_id,))
    conn.commit()


def get_cat(conn: sqlite3.Connection, name: str) -> Cat:
    row = conn.execute("SELECT * FROM cats WHERE name = ?", (name,)).fetchone()
    if row is None:
        raise KeyError(f"no cat named {name!r}")
    taboos = tuple(
        r["ingredient_key"] for r in conn.execute(
            "SELECT ingredient_key FROM cat_taboos WHERE cat_id = ?", (row["id"],)
        )
    )
    return Cat(
        id=row["id"], name=row["name"], dob=row["dob"],
        weight_kg=row["weight_kg"], activity=row["activity"],
        texture_max=row["texture_max"], notes=row["notes"],
        taboos=taboos,
    )


def list_cats(conn: sqlite3.Connection) -> list[Cat]:
    return [get_cat(conn, r["name"]) for r in conn.execute("SELECT name FROM cats ORDER BY name")]


def record_feeding(
    conn: sqlite3.Connection,
    *,
    cat_id: int,
    recipe: dict[str, float],
    response: str,
    served_at: datetime | None = None,
) -> int:
    if response not in VALID_RESPONSES:
        raise ValueError(f"response must be one of {VALID_RESPONSES}")
    ts = (served_at or datetime.now()).isoformat(timespec="seconds")
    cur = conn.execute(
        "INSERT INTO feedings(cat_id, served_at, recipe_json, response) "
        "VALUES (?, ?, ?, ?)",
        (cat_id, ts, json.dumps(recipe, sort_keys=True), response),
    )
    conn.commit()
    return int(cur.lastrowid)


def feeding_history(
    conn: sqlite3.Connection, cat_id: int, limit: int = 20
) -> list[dict]:
    rows = conn.execute(
        "SELECT id, served_at, recipe_json, response "
        "FROM feedings WHERE cat_id = ? ORDER BY served_at DESC, id DESC LIMIT ?",
        (cat_id, limit),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "served_at": r["served_at"],
            "recipe": json.loads(r["recipe_json"]),
            "response": r["response"],
        }
        for r in rows
    ]
