"""Three-screen Tkinter GUI for the cat-food recipe system.

Tabs:
  1. Profiles    — list, add, edit, delete cats; pick taboos from the
                   ingredient catalogue; set DOB / weight / texture cap.
  2. Generate    — pick a cat, generate top-N candidate recipes, hit a
                   feedback button to record outcome (which also nudges
                   the cat's preference weights).
  3. History     — recent feedings per cat.

Heavy lifting (DB, recommender, weights, ML model) lives in the
underlying modules; this file is the view+controller layer.

Run:
    py -m src.gui.app
"""
from __future__ import annotations

import sqlite3
from datetime import date
from pathlib import Path
from tkinter import (
    BOTH, END, EXTENDED, LEFT, RIGHT, TOP, X, Y,
    StringVar, Tk, messagebox,
)
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText

import joblib
import pandas as pd

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


# --- helpers -----------------------------------------------------------

def _parse_dob(s: str) -> date:
    return date.fromisoformat(s.strip())


def _format_recipe(recipe: dict[str, float], ingredients: pd.DataFrame) -> str:
    parts = []
    for k, g in sorted(recipe.items(), key=lambda kv: -kv[1]):
        display = ingredients.loc[k, "display"] if k in ingredients.index else k
        parts.append(f"{display} {g:.1f}g")
    return " · ".join(parts)


# --- profile tab -------------------------------------------------------

class ProfileTab(ttk.Frame):
    def __init__(self, master, app: "CatFoodApp"):
        super().__init__(master, padding=10)
        self.app = app
        self.selected_cat_id: int | None = None

        # Left: list of cats
        left = ttk.Frame(self)
        left.pack(side=LEFT, fill=Y, padx=(0, 12))
        ttk.Label(left, text="Cats", font=("", 10, "bold")).pack(anchor="w")
        self.cat_list = ttk.Treeview(left, columns=("name",), show="headings", height=14)
        self.cat_list.heading("name", text="Name")
        self.cat_list.column("name", width=160)
        self.cat_list.pack(fill=Y)
        self.cat_list.bind("<<TreeviewSelect>>", self._on_select)

        btns = ttk.Frame(left)
        btns.pack(fill=X, pady=(8, 0))
        ttk.Button(btns, text="New",    command=self._on_new).pack(side=LEFT)
        ttk.Button(btns, text="Delete", command=self._on_delete).pack(side=LEFT, padx=4)

        # Right: form
        right = ttk.Frame(self)
        right.pack(side=LEFT, fill=BOTH, expand=True)

        form = ttk.Frame(right)
        form.pack(fill=X)

        self.name_var      = StringVar()
        self.dob_var       = StringVar()
        self.weight_var    = StringVar()
        self.activity_var  = StringVar(value="medium")
        self.texture_var   = StringVar(value="hard")
        self.notes_var     = StringVar()

        rows = [
            ("Name",         ttk.Entry(form, textvariable=self.name_var, width=24)),
            ("DOB (YYYY-MM-DD)", ttk.Entry(form, textvariable=self.dob_var, width=14)),
            ("Weight (kg)",  ttk.Entry(form, textvariable=self.weight_var, width=8)),
            ("Activity",     ttk.Combobox(form, textvariable=self.activity_var,
                                          values=ACTIVITIES, state="readonly", width=10)),
            ("Texture max",  ttk.Combobox(form, textvariable=self.texture_var,
                                          values=TEXTURES, state="readonly", width=10)),
            ("Notes",        ttk.Entry(form, textvariable=self.notes_var, width=40)),
        ]
        for i, (label, widget) in enumerate(rows):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=2)
            widget.grid(row=i, column=1, sticky="w", pady=2)

        ttk.Label(right, text="Taboos (Ctrl/Shift to multi-select):",
                  font=("", 10, "bold")).pack(anchor="w", pady=(12, 2))
        taboo_wrap = ttk.Frame(right)
        taboo_wrap.pack(fill=BOTH, expand=True)
        from tkinter import Listbox
        self.taboo_box = Listbox(taboo_wrap, selectmode=EXTENDED, height=12,
                                 exportselection=False)
        sb = ttk.Scrollbar(taboo_wrap, orient="vertical",
                           command=self.taboo_box.yview)
        self.taboo_box.configure(yscrollcommand=sb.set)
        self.taboo_box.pack(side=LEFT, fill=BOTH, expand=True)
        sb.pack(side=RIGHT, fill=Y)

        # Populate taboo options from the ingredient catalogue
        self._taboo_keys: list[str] = []
        self.reload_taboo_options()

        save = ttk.Button(right, text="Save", command=self._on_save)
        save.pack(anchor="e", pady=(10, 0))

        self.refresh()

    def refresh(self) -> None:
        for iid in self.cat_list.get_children():
            self.cat_list.delete(iid)
        for c in list_cats(self.app.conn):
            self.cat_list.insert("", END, iid=str(c.id), values=(c.name,))

    def reload_taboo_options(self) -> None:
        """Rebuild the taboo Listbox from the latest merged catalogue.

        Called once at construction and again whenever the Library tab
        adds or deletes a custom ingredient.
        """
        self.taboo_box.delete(0, END)
        self._taboo_keys.clear()
        for key, row in self.app.ingredients.iterrows():
            self._taboo_keys.append(key)
            label = f"{row['display']}  ({row['category']})"
            if bool(row.get("custom", False)):
                label = "* " + label
            self.taboo_box.insert(END, label)

    def _on_select(self, _ev):
        sel = self.cat_list.selection()
        if not sel:
            return
        cat = get_cat(self.app.conn, self.cat_list.item(sel[0])["values"][0])
        self.selected_cat_id = cat.id
        self.name_var.set(cat.name)
        self.dob_var.set(cat.dob)
        self.weight_var.set(str(cat.weight_kg))
        self.activity_var.set(cat.activity)
        self.texture_var.set(cat.texture_max)
        self.notes_var.set(cat.notes or "")
        self.taboo_box.selection_clear(0, END)
        for i, k in enumerate(self._taboo_keys):
            if k in cat.taboos:
                self.taboo_box.selection_set(i)

    def _on_new(self):
        self.selected_cat_id = None
        for v in (self.name_var, self.dob_var, self.weight_var, self.notes_var):
            v.set("")
        self.activity_var.set("medium")
        self.texture_var.set("hard")
        self.taboo_box.selection_clear(0, END)
        self.cat_list.selection_remove(self.cat_list.selection())

    def _on_save(self):
        try:
            name = self.name_var.get().strip()
            if not name:
                raise ValueError("name is required")
            dob = _parse_dob(self.dob_var.get())
            weight = float(self.weight_var.get())
            taboos = [self._taboo_keys[i] for i in self.taboo_box.curselection()]
        except (ValueError, KeyError) as e:
            messagebox.showerror("Invalid input", str(e))
            return

        try:
            if self.selected_cat_id is None:
                add_cat(
                    self.app.conn,
                    name=name, dob=dob, weight_kg=weight,
                    activity=self.activity_var.get(),
                    texture_max=self.texture_var.get(),
                    notes=self.notes_var.get(),
                    taboos=taboos,
                )
            else:
                update_cat(
                    self.app.conn, cat_id=self.selected_cat_id,
                    weight_kg=weight, activity=self.activity_var.get(),
                    texture_max=self.texture_var.get(),
                    notes=self.notes_var.get(), taboos=taboos,
                )
        except sqlite3.IntegrityError as e:
            messagebox.showerror("DB error", str(e))
            return

        self.refresh()
        self.app.broadcast_cats_changed()
        messagebox.showinfo("Saved", f"Profile for {name} saved.")

    def _on_delete(self):
        if self.selected_cat_id is None:
            return
        if not messagebox.askyesno("Delete", "Delete this cat and all their feedings?"):
            return
        delete_cat(self.app.conn, self.selected_cat_id)
        self._on_new()
        self.refresh()
        self.app.broadcast_cats_changed()


# --- generate tab ------------------------------------------------------

class GenerateTab(ttk.Frame):
    def __init__(self, master, app: "CatFoodApp"):
        super().__init__(master, padding=10)
        self.app = app

        top = ttk.Frame(self)
        top.pack(fill=X)
        ttk.Label(top, text="Cat:").pack(side=LEFT)
        self.cat_var = StringVar()
        self.cat_combo = ttk.Combobox(top, textvariable=self.cat_var,
                                      state="readonly", width=20)
        self.cat_combo.pack(side=LEFT, padx=4)

        ttk.Label(top, text="Top:").pack(side=LEFT, padx=(12, 0))
        self.top_var = StringVar(value="5")
        ttk.Spinbox(top, from_=1, to=10, width=4,
                    textvariable=self.top_var).pack(side=LEFT, padx=4)

        ttk.Button(top, text="Generate", command=self._on_generate).pack(side=LEFT, padx=12)

        self.results = ScrolledText(self, height=22, wrap="word", font=("Consolas", 10))
        self.results.pack(fill=BOTH, expand=True, pady=(10, 0))
        self.results.configure(state="disabled")

        self.actions_frame = ttk.Frame(self)
        self.actions_frame.pack(fill=X, pady=(8, 0))
        ttk.Label(self.actions_frame, text="After serving recipe #:").pack(side=LEFT)
        self.pick_var = StringVar(value="1")
        ttk.Spinbox(self.actions_frame, from_=1, to=10, width=4,
                    textvariable=self.pick_var).pack(side=LEFT, padx=4)
        for resp in RESPONSES:
            ttk.Button(
                self.actions_frame, text=resp,
                command=lambda r=resp: self._on_feedback(r),
            ).pack(side=LEFT, padx=4)

        self._last_ranked: list = []
        self._last_cat_id: int | None = None

        self.refresh_cats()

    def refresh_cats(self) -> None:
        names = [c.name for c in list_cats(self.app.conn)]
        self.cat_combo["values"] = names
        if names and not self.cat_var.get():
            self.cat_var.set(names[0])

    def _set_results(self, text: str) -> None:
        self.results.configure(state="normal")
        self.results.delete("1.0", END)
        self.results.insert(END, text)
        self.results.configure(state="disabled")

    def _on_generate(self):
        name = self.cat_var.get()
        if not name:
            messagebox.showinfo("No cat", "Add a cat in the Profiles tab first.")
            return
        try:
            top_n = max(1, int(self.top_var.get()))
        except ValueError:
            top_n = 5

        try:
            cat = get_cat(self.app.conn, name)
            age = compute_age_years(date.fromisoformat(cat.dob))
            pool = applicable_ingredients(
                self.app.ingredients, taboos=cat.taboos, texture_max=cat.texture_max,
            )
            cands = propose(pool, n_candidates=300, seed=None)
            weights = get_weights(self.app.conn, cat.id)
            ranked = rank(
                cands, model=self.app.model, age_years=age,
                weights=weights, top_n=top_n, ingredients=self.app.ingredients,
            )
        except Exception as e:
            messagebox.showerror("Generate failed", str(e))
            return

        self._last_ranked = ranked
        self._last_cat_id = cat.id

        lines = [f"Top {len(ranked)} for {cat.name} (age {age:.1f}y):\n"]
        for i, r in enumerate(ranked, 1):
            tag = "rule" if r.used_rule_scorer else "ml"
            lines.append(
                f"[{i}] blended {r.blended:5.1f}  {tag} {r.ml_score:5.1f}  "
                f"pref x{r.pref_mean:.2f}"
            )
            lines.append(f"    {_format_recipe(r.recipe, self.app.ingredients)}")
            lines.append("")
        self._set_results("\n".join(lines))

    def _on_feedback(self, response: str):
        if not self._last_ranked:
            messagebox.showinfo("No recipe", "Generate recipes first.")
            return
        try:
            pick = int(self.pick_var.get())
        except ValueError:
            pick = 1
        if not (1 <= pick <= len(self._last_ranked)):
            messagebox.showerror("Bad pick",
                                 f"Pick must be 1..{len(self._last_ranked)}")
            return
        recipe = self._last_ranked[pick - 1].recipe
        record_feeding(self.app.conn, cat_id=self._last_cat_id,
                       recipe=recipe, response=response)
        new_w = update_after_feeding(self.app.conn, cat_id=self._last_cat_id,
                                     recipe=recipe, response=response)
        changed = ", ".join(f"{k}:{new_w[k]:.2f}" for k in recipe)
        messagebox.showinfo("Recorded",
                            f"Logged '{response}' for recipe #{pick}.\n\n"
                            f"Updated weights: {changed}")
        self.app.broadcast_history_changed()


# --- history tab -------------------------------------------------------

class HistoryTab(ttk.Frame):
    def __init__(self, master, app: "CatFoodApp"):
        super().__init__(master, padding=10)
        self.app = app

        top = ttk.Frame(self)
        top.pack(fill=X)
        ttk.Label(top, text="Cat:").pack(side=LEFT)
        self.cat_var = StringVar()
        self.cat_combo = ttk.Combobox(top, textvariable=self.cat_var,
                                      state="readonly", width=20)
        self.cat_combo.pack(side=LEFT, padx=4)
        self.cat_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh_history())
        ttk.Button(top, text="Refresh", command=self.refresh_history).pack(side=LEFT, padx=8)

        cols = ("id", "served_at", "response", "recipe")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=20)
        for c, w in zip(cols, (40, 150, 80, 700)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill=BOTH, expand=True, pady=(10, 0))

        self.refresh_cats()

    def refresh_cats(self) -> None:
        names = [c.name for c in list_cats(self.app.conn)]
        self.cat_combo["values"] = names
        if names and not self.cat_var.get():
            self.cat_var.set(names[0])

    def refresh_history(self) -> None:
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        if not self.cat_var.get():
            return
        cat = get_cat(self.app.conn, self.cat_var.get())
        for row in feeding_history(self.app.conn, cat.id, limit=50):
            self.tree.insert(
                "", END,
                values=(
                    row["id"], row["served_at"], row["response"],
                    _format_recipe(row["recipe"], self.app.ingredients),
                ),
            )


# --- library tab -------------------------------------------------------

class LibraryTab(ttk.Frame):
    """Browse the merged ingredient catalogue and add custom items.

    Custom ingredients are stored in data/processed/custom_ingredients.csv
    (separate from the frozen base CSV). They appear in recipe generation
    immediately, and recipes containing them are scored via the rule
    scorer instead of the ML model — see rank.py.
    """
    def __init__(self, master, app: "CatFoodApp"):
        super().__init__(master, padding=10)
        self.app = app

        # Top: catalogue table
        ttk.Label(self, text="Catalogue (custom items prefixed with *):",
                  font=("", 10, "bold")).pack(anchor="w")
        cols = ("key", "display", "category", "texture", "kcal", "protein_g", "fat_g")
        self.tree = ttk.Treeview(self, columns=cols, show="headings", height=12)
        for c, w in zip(cols, (160, 220, 90, 80, 70, 80, 70)):
            self.tree.heading(c, text=c)
            self.tree.column(c, width=w, anchor="w")
        self.tree.pack(fill=BOTH, expand=True, pady=(4, 4))
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        del_btn = ttk.Button(self, text="Delete selected (custom only)",
                             command=self._on_delete)
        del_btn.pack(anchor="w", pady=(0, 8))

        # Bottom: add form
        sep = ttk.Separator(self, orient="horizontal")
        sep.pack(fill=X, pady=4)
        ttk.Label(self, text="Add a custom ingredient:",
                  font=("", 10, "bold")).pack(anchor="w", pady=(4, 4))

        form = ttk.Frame(self)
        form.pack(fill=X)

        self.key_var      = StringVar()
        self.display_var  = StringVar()
        self.category_var = StringVar(value="protein")
        self.texture_var  = StringVar(value="soft")

        meta_rows = [
            ("Key (lowercase, no spaces)",
             ttk.Entry(form, textvariable=self.key_var, width=22)),
            ("Display name",
             ttk.Entry(form, textvariable=self.display_var, width=28)),
            ("Category",
             ttk.Combobox(form, textvariable=self.category_var,
                          values=VALID_CATEGORIES, state="readonly", width=14)),
            ("Texture",
             ttk.Combobox(form, textvariable=self.texture_var,
                          values=VALID_TEXTURES, state="readonly", width=14)),
        ]
        for i, (label, w) in enumerate(meta_rows):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", pady=2, padx=(0, 8))
            w.grid(row=i, column=1, sticky="w", pady=2)

        # Nutrient grid (per 100g as-fed)
        ttk.Label(form, text="Per 100g as-fed nutrients:",
                  font=("", 9, "italic")).grid(row=0, column=2, sticky="w",
                                                padx=(20, 4), pady=2)
        self.nutrient_vars: dict[str, StringVar] = {}
        for j, col in enumerate(NUTRIENT_COLUMNS):
            v = StringVar(value="0")
            self.nutrient_vars[col] = v
            r = (j % 6) + 1
            c = 2 + (j // 6) * 2
            ttk.Label(form, text=col).grid(row=r, column=c, sticky="w",
                                            padx=(20, 4), pady=1)
            ttk.Entry(form, textvariable=v, width=10).grid(
                row=r, column=c + 1, sticky="w", pady=1
            )

        ttk.Button(self, text="Add to library",
                   command=self._on_add).pack(anchor="e", pady=(10, 0))

        self.refresh()

    def refresh(self) -> None:
        for iid in self.tree.get_children():
            self.tree.delete(iid)
        custom = list_custom_keys()
        df = load_ingredients()
        for key, row in df.iterrows():
            prefix = "* " if key in custom else "  "
            self.tree.insert(
                "", END, iid=key,
                values=(
                    prefix + key, row.get("display", ""),
                    row.get("category", ""), row.get("texture", ""),
                    f"{row.get('kcal', 0):.0f}",
                    f"{row.get('protein_g', 0):.1f}",
                    f"{row.get('fat_g', 0):.1f}",
                ),
            )

    def _on_select(self, _ev):
        pass  # selection drives the Delete button — nothing else for now

    def _on_delete(self):
        sel = self.tree.selection()
        if not sel:
            return
        key = sel[0]
        if key not in list_custom_keys():
            messagebox.showinfo("Built-in", "Cannot delete a built-in ingredient.")
            return
        if not messagebox.askyesno("Delete", f"Remove custom ingredient {key!r}?"):
            return
        delete_custom(key)
        self.refresh()
        self.app.broadcast_ingredients_changed()

    def _on_add(self):
        try:
            nutrients = {k: float(v.get() or 0) for k, v in self.nutrient_vars.items()}
            add_custom(
                key=self.key_var.get().strip(),
                display=self.display_var.get(),
                category=self.category_var.get(),
                texture=self.texture_var.get(),
                nutrients=nutrients,
            )
        except (ValueError, KeyError) as e:
            messagebox.showerror("Invalid input", str(e))
            return

        # Clear the form, refresh the catalogue, and tell the rest of
        # the app that the ingredient pool changed.
        self.key_var.set("")
        self.display_var.set("")
        for v in self.nutrient_vars.values():
            v.set("0")
        self.refresh()
        self.app.broadcast_ingredients_changed()
        messagebox.showinfo("Added", "Custom ingredient saved to library.")


# --- shell -------------------------------------------------------------

class CatFoodApp(Tk):
    def __init__(self):
        super().__init__()
        self.title("Cat Food Recipe System")
        self.geometry("1080x680")

        # Lazy-load shared resources up front so the UI never stalls later.
        if not INGREDIENTS_CSV.exists():
            messagebox.showerror("Missing data",
                                 f"{INGREDIENTS_CSV} not found.\n"
                                 "Run `py -m src.data.build_dataset` first.")
            self.destroy()
            return
        if not MODEL_PATH.exists():
            messagebox.showerror("Missing model",
                                 f"{MODEL_PATH} not found.\n"
                                 "Run `py -m src.ml.train_alt` first.")
            self.destroy()
            return

        self.ingredients = load_ingredients()
        self.model = joblib.load(MODEL_PATH)
        self.conn = connect()

        nb = ttk.Notebook(self)
        nb.pack(fill=BOTH, expand=True)
        self.profile_tab  = ProfileTab(nb, self)
        self.generate_tab = GenerateTab(nb, self)
        self.history_tab  = HistoryTab(nb, self)
        self.library_tab  = LibraryTab(nb, self)
        nb.add(self.profile_tab,  text="Profiles")
        nb.add(self.generate_tab, text="Generate")
        nb.add(self.history_tab,  text="History")
        nb.add(self.library_tab,  text="Library")
        nb.bind("<<NotebookTabChanged>>", self._on_tab_change)
        self._nb = nb

    def broadcast_cats_changed(self) -> None:
        """Profile tab tells the others a cat list might have changed."""
        self.generate_tab.refresh_cats()
        self.history_tab.refresh_cats()

    def broadcast_history_changed(self) -> None:
        """Generate tab tells History a new feeding landed."""
        self.history_tab.refresh_history()

    def broadcast_ingredients_changed(self) -> None:
        """Library tab tells the rest of the app that the catalogue grew/shrunk.

        The Profile tab's taboo Listbox is rebuilt from scratch so newly
        added ingredients are selectable as taboos. The Generate tab and
        recommender pull from `self.app.ingredients` at call time, so we
        just refresh the cached frame here.
        """
        self.ingredients = load_ingredients()
        # ProfileTab caches taboo options at construction time → rebuild it.
        self.profile_tab.reload_taboo_options()

    def _on_tab_change(self, _ev) -> None:
        current = self._nb.select()
        if current == str(self.history_tab):
            self.history_tab.refresh_history()


def main() -> None:
    app = CatFoodApp()
    app.mainloop()


if __name__ == "__main__":
    main()
