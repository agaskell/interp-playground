"""Backend._pool without a model: Neuronpedia failures must be loud, not silent."""

import pytest
import torch

from interp_playground import backend as backend_module
from interp_playground.backend import Backend


@pytest.fixture
def bare_backend(monkeypatch):
    b = Backend()
    monkeypatch.setattr(Backend, "_prepare_layer", lambda self, layer: None)
    zeros = torch.zeros(backend_module.D_SAE, dtype=torch.bool)
    b._hidden[12] = {"dense": zeros.clone(), "position_artifact": zeros.clone()}
    return b


def test_pool_raises_when_a_random_feature_fetch_fails(bare_backend, monkeypatch):
    calls = []

    def fake_load(layer, feature):
        calls.append(feature)
        if feature == 7272:
            return [{"max": 1.0, "tokens": [f"▁w{i}" for i in range(20)], "peak": 15}] * 10
        return None  # Neuronpedia unreachable for every other feature

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", fake_load)
    with pytest.raises(ValueError, match="Neuronpedia .*feature"):
        bare_backend._pool(7272, 12, seed=0, n_heldout=2, n_random=8)
    # Short-circuits after the first failure: one own fetch plus one failed random fetch.
    assert len(calls) == 2


def test_pool_raises_when_own_feature_fetch_fails(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", lambda layer, feature: None)
    with pytest.raises(ValueError, match="Neuronpedia .* 7272"):
        bare_backend._pool(7272, 12, seed=0, n_heldout=2, n_random=0)


def test_pool_is_deterministic_and_short_only_when_examples_are_few(bare_backend, monkeypatch):
    def fake_load(layer, feature):
        n = 9 if feature == 7272 else 3  # own feature: only one held-out example beyond the 8 shown
        return [{"max": 1.0, "tokens": [f"▁f{feature}t{i}" for i in range(20)], "peak": 15}] * n

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", fake_load)
    a = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=3)
    b = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=3)
    assert a == b
    assert [p["source"] for p in a].count("heldout") == 1
    assert [p["source"] for p in a].count("random") == 3


def test_pool_heldout_spans_the_range_and_peak_position_varies(bare_backend, monkeypatch):
    def fake_load(layer, feature):
        n = 45 if feature == 7272 else 3
        return [{"max": 1.0, "tokens": [f"▁e{i}t{j}" for j in range(40)], "peak": 20} for i in range(n)]

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", fake_load)
    pool = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=6)
    heldout = [p["text"] for p in pool if p["source"] == "heldout"]
    assert len(heldout) == 4
    first_words = {t.split()[0].split("t")[0] for t in heldout}
    assert "e8" in first_words and "e44" in first_words  # spread from the 9th to the last example
    last_offsets = {int(p["text"].split()[-1].split("t")[1]) for p in pool}
    assert len(last_offsets) > 1  # the peak is not always the same distance from the end
