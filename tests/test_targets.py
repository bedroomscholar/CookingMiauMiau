from datetime import date

import pytest

from src.nutrition.targets import (
    compute_age_years,
    stage_for,
    targets_for,
    targets_for_dob,
)


def test_age_zero_on_birthday():
    assert compute_age_years(date(2020, 5, 3), today=date(2020, 5, 3)) == 0.0


def test_age_exactly_one_year():
    assert compute_age_years(date(2020, 5, 3), today=date(2021, 5, 3)) == pytest.approx(1.0)


def test_age_fraction_six_months():
    age = compute_age_years(date(2020, 5, 3), today=date(2020, 11, 3))
    assert 0.4 < age < 0.6


def test_age_just_before_anniversary():
    age = compute_age_years(date(2020, 5, 3), today=date(2026, 5, 2))
    assert 5.99 < age < 6.0


def test_dob_in_future_raises():
    with pytest.raises(ValueError):
        compute_age_years(date(2030, 1, 1), today=date(2026, 5, 3))


@pytest.mark.parametrize(
    "age,expected",
    [
        (0.0, "kitten"),
        (0.99, "kitten"),
        (1.0, "adult"),
        (6.99, "adult"),
        (7.0, "mature"),
        (9.99, "mature"),
        (10.0, "senior"),
        (18.0, "senior"),
    ],
)
def test_stage_boundaries(age, expected):
    assert stage_for(age) == expected


def test_cats_at_ten_are_senior_stage():
    """Changcheng & Delong, both ~10y, fall on the senior side of the 10.0 cutoff."""
    stage, bands = targets_for_dob(date(2016, 5, 3), today=date(2026, 5, 3))
    assert stage == "senior"
    assert bands["protein_g"].min == 32.0


def test_senior_phosphorus_is_capped_below_mature():
    assert targets_for(11.5)["phosphorus_mg"].max < targets_for(8.0)["phosphorus_mg"].max
