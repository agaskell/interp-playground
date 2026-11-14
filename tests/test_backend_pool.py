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
    # No model in these tests: every window fires unless a test says otherwise.
    monkeypatch.setattr(Backend, "_window_fires", lambda self, feature, layer, text: True)
    return b


def distinct_examples(feature, n):
    return [{"max": 1.0, "tokens": [f"▁e{i}t{j}" for j in range(40)], "peak": 20} for i in range(n)]


def example_ids(pool, source="heldout"):
    """Which example index each pool sentence came from (tokens are e<i>t<j>)."""
    return sorted(int(p["text"].split()[0][1:].split("t")[0]) for p in pool if p["source"] == source)


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
        if feature == 7272:
            return distinct_examples(feature, 11)  # only three held-out examples beyond the 8 shown
        return [{"max": 1.0, "tokens": [f"▁f{feature}t{i}" for i in range(20)], "peak": 15}] * 3

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", fake_load)
    a = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=3)
    b = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=3)
    assert a == b
    assert [p["source"] for p in a].count("heldout") == 3
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


def load_45(layer, feature):
    return distinct_examples(feature, 45 if feature == 7272 else 3)


def test_pool_replaces_heldout_windows_that_do_not_fire(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    silent = {"e8", "e20"}  # two of the four spread picks (8, 20, 32, 44) are silent in a short window
    monkeypatch.setattr(
        Backend, "_window_fires",
        lambda self, feature, layer, text: text.split()[0].split("t")[0] not in silent,
    )
    pool = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert example_ids(pool) == [9, 21, 32, 44]  # each silent pick replaced by its nearest firing neighbour


def test_pool_refuses_a_feature_whose_heldout_windows_do_not_fire(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    fires = {"e10", "e11"}
    monkeypatch.setattr(
        Backend, "_window_fires",
        lambda self, feature, layer, text: text.split()[0].split("t")[0] in fires,
    )
    with pytest.raises(ValueError, match="only 2 .* fire"):
        bare_backend._pool(7272, 12, seed=0, n_heldout=6, n_random=2)
    # Asking for fewer held-out sentences than the floor only requires that many.
    assert example_ids(bare_backend._pool(7272, 12, seed=0, n_heldout=2, n_random=2)) == [10, 11]


def test_pool_is_built_once_per_key(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    checks = []
    monkeypatch.setattr(
        Backend, "_window_fires", lambda self, feature, layer, text: checks.append(text) or True
    )
    a = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    n = len(checks)
    b = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert a == b and len(checks) == n == 4
    a[0]["text"] = "mutated"  # callers get copies, the cache is not theirs to change
    assert bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2) == b
