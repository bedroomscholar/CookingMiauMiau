import pandas as pd
import pytest

from src.nutrition.scorer import _band_score, per_100g_dm, score
from src.nutrition.targets import NutrientBand


def make_ingredients() -> pd.DataFrame:
    """Three synthetic ingredients with hand-picked numbers for predictable tests."""
    rows = [
        # key, moisture, protein, fat, ca, p, taurine, mg, na, k
        ("meat",       70.0, 25.0, 8.0,  10.0, 200.0, 80.0,  25.0, 80.0, 350.0),
        ("organ",      72.0, 18.0, 5.0, 100.0, 350.0, 120.0, 22.0, 90.0, 300.0),
        ("supplement",  0.0,  0.0, 0.0, 4000.0,  0.0,  500.0, 0.0,  0.0,   0.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "key", "moisture_g", "protein_g", "fat_g",
            "calcium_mg", "phosphorus_mg", "taurine_mg",
            "magnesium_mg", "sodium_mg", "potassium_mg",
        ],
    ).set_index("key")
    return df


def test_band_score_inside_band_is_100():
    band = NutrientBand(min=10, target=20, max=30, unit="x")
    assert _band_score(10, band) == 100
    assert _band_score(20, band) == 100
    assert _band_score(30, band) == 100


def test_band_score_floor_and_ceiling_are_zero():
    band = NutrientBand(min=10, target=20, max=30, unit="x")
    assert _band_score(5, band) == 0   # 50% below min
    assert _band_score(45, band) == 0  # 50% above max


def test_band_score_decays_linearly():
    band = NutrientBand(min=10, target=20, max=30, unit="x")
    # halfway between floor (5) and min (10) → 50
    assert _band_score(7.5, band) == pytest.approx(50.0)
    # halfway between max (30) and ceil (45) → 50
    assert _band_score(37.5, band) == pytest.approx(50.0)


def test_per_100g_dm_basic_math():
    ing = make_ingredients()
    # 100g meat (70g water → 30g DM, 25g protein) = 25/30*100 = 83.33 g protein/100g DM
    out = per_100g_dm({"meat": 100.0}, ing)
    assert out["protein_g"] == pytest.approx(83.333, rel=1e-3)
    assert out["calcium_mg"] == pytest.approx(10 / 30 * 100, rel=1e-3)
    assert out["ca_p_ratio"] == pytest.approx(10 / 200, rel=1e-3)


def test_per_100g_dm_supplement_pulls_up_calcium():
    ing = make_ingredients()
    out = per_100g_dm({"meat": 100.0, "supplement": 1.0}, ing)
    # Adding 1g supplement (0g moisture, 40mg Ca) to 100g meat (30g DM)
    # total DM = 30 + 1 = 31; total Ca = 10 + 40 = 50 → 50/31*100 ≈ 161
    assert out["calcium_mg"] == pytest.approx(50 / 31 * 100, rel=1e-3)


def test_unknown_ingredient_raises():
    ing = make_ingredients()
    with pytest.raises(KeyError):
        per_100g_dm({"ghost": 50.0}, ing)


def test_empty_recipe_raises():
    ing = make_ingredients()
    with pytest.raises(ValueError):
        per_100g_dm({}, ing)


def test_score_returns_overall_and_subscores():
    ing = make_ingredients()
    rep = score({"meat": 100.0, "organ": 30.0, "supplement": 0.5}, ing, age_years=10.0)
    assert 0 <= rep.overall <= 100
    assert "protein_g" in rep.per_nutrient
    assert "ca_p_ratio" in rep.per_nutrient
    assert all(0 <= v <= 100 for v in rep.per_nutrient.values())
