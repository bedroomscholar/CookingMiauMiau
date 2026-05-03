"""Mocked USDA client tests — never hit the real API."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import responses

from src.data.usda_client import API_URL, UsdaClient

SAMPLE = {
    "fdcId": 12345,
    "description": "Sample chicken thigh",
    "foodNutrients": [
        {"nutrient": {"id": 1003, "name": "Protein"},   "amount": 18.5},
        {"nutrient": {"id": 1004, "name": "Total fat"}, "amount": 8.1},
        {"nutrient": {"id": 1051, "name": "Water"},     "amount": 71.0},
        {"nutrient": {"id": 1087, "name": "Calcium"},   "amount": 11.0},
        {"nutrient": {"id": 1091, "name": "Phosphorus"},"amount": 175.0},
        # taurine omitted on purpose — should default to 0
    ],
}


@pytest.fixture
def tmp_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> UsdaClient:
    monkeypatch.setenv("USDA_API_KEY", "fake-key-for-tests")
    return UsdaClient(cache_dir=tmp_path / "raw", cache_only=False)


@responses.activate
def test_first_call_hits_network_and_caches(tmp_client: UsdaClient):
    responses.add(responses.GET, API_URL.format(fdc_id=12345), json=SAMPLE, status=200)
    out = tmp_client.get_nutrients(12345)
    assert out["protein_g"] == 18.5
    assert out["moisture_g"] == 71.0
    assert out["taurine_mg"] == 0.0  # absent → default 0
    assert (tmp_client.cache_dir / "12345.json").exists()
    assert len(responses.calls) == 1


@responses.activate
def test_second_call_uses_cache(tmp_client: UsdaClient):
    responses.add(responses.GET, API_URL.format(fdc_id=12345), json=SAMPLE, status=200)
    tmp_client.get_nutrients(12345)
    tmp_client.get_nutrients(12345)
    assert len(responses.calls) == 1


def test_cache_only_uses_disk(tmp_path: Path):
    cache = tmp_path / "raw"
    cache.mkdir()
    (cache / "12345.json").write_text(json.dumps(SAMPLE), encoding="utf-8")
    client = UsdaClient(cache_dir=cache, cache_only=True)
    out = client.get_nutrients(12345)
    assert out["protein_g"] == 18.5


def test_cache_only_misses_raise(tmp_path: Path):
    client = UsdaClient(cache_dir=tmp_path / "raw", cache_only=True)
    with pytest.raises(RuntimeError, match="not cached"):
        client.get_raw(99999)
