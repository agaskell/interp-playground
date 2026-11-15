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
    monkeypatch.setattr(Backend, "_window_fires", lambda self, feature, layer, text, recorded=None: True)
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
            return distinct_examples(feature, 10)  # distinct texts: copies of shown examples are not held out
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
        lambda self, feature, layer, text, recorded=None: text.split()[0].split("t")[0] not in silent,
    )
    pool = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert example_ids(pool) == [9, 21, 32, 44]  # each silent pick replaced by its nearest firing neighbour


def test_pool_refuses_a_feature_whose_heldout_windows_do_not_fire(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    fires = {"e10", "e11"}
    monkeypatch.setattr(
        Backend, "_window_fires",
        lambda self, feature, layer, text, recorded=None: text.split()[0].split("t")[0] in fires,
    )
    with pytest.raises(ValueError, match="only 2 .* keep half"):
        bare_backend._pool(7272, 12, seed=0, n_heldout=6, n_random=2)
    # Asking for fewer held-out sentences than the floor only requires that many.
    assert example_ids(bare_backend._pool(7272, 12, seed=0, n_heldout=2, n_random=2)) == [10, 11]


def test_pool_is_built_once_per_key(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    checks = []
    monkeypatch.setattr(
        Backend, "_window_fires", lambda self, feature, layer, text, recorded=None: checks.append(text) or True
    )
    a = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    n = len(checks)
    b = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert a == b and len(checks) == n == 4
    a[0]["text"] = "mutated"  # callers get copies, the cache is not theirs to change
    assert bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2) == b


def test_pool_never_holds_out_an_example_feature_info_shows(bare_backend, monkeypatch):
    def fake_load(layer, feature):
        ex = distinct_examples(feature, 45 if feature == 7272 else 3)
        if feature == 7272:
            ex[8] = dict(ex[2])   # Neuronpedia repeats examples: index 8 is a copy of one of the 8 shown
            ex[20] = dict(ex[5])
        return ex

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", fake_load)
    pool = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert example_ids(pool) == [9, 21, 32, 44]  # the copies are skipped like silent windows


def test_eval_sentences_reports_the_pool_version(bare_backend, monkeypatch):
    from interp_playground.evals import POOL_VERSION

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    out = bare_backend.eval_sentences(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert out["pool_version"] == POOL_VERSION
    assert all(set(s) == {"id", "text"} for s in out["sentences"])


def test_pool_skips_a_window_with_no_text(bare_backend, monkeypatch):
    def fake_load(layer, feature):
        ex = distinct_examples(feature, 45 if feature == 7272 else 3)
        if feature == 7272:
            ex[8] = {"max": 1.0, "tokens": ["▁"] * 40, "peak": 20}  # whitespace only: nothing to run
        return ex

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", fake_load)
    pool = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert all(p["text"].strip() for p in pool)
    assert example_ids(pool) == [9, 20, 32, 44]


# --- pool version 5 (DECISIONS 2026-11-14, after run 4) ----------------------

def test_heldout_windows_are_checked_against_the_recorded_activation(bare_backend, monkeypatch):
    def fake_load(layer, feature):
        ex = distinct_examples(feature, 45 if feature == 7272 else 3)
        for i, e in enumerate(ex):
            e["max"] = 100.0 - i  # what Neuronpedia recorded for this example in its full context
        return ex

    seen = {}

    def fires(self, feature, layer, text, recorded=None):
        seen[int(text.split()[0][1:].split("t")[0])] = recorded
        return True

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", fake_load)
    monkeypatch.setattr(Backend, "_window_fires", fires)
    bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)
    assert seen == {8: 92.0, 20: 80.0, 32: 68.0, 44: 56.0}


def scanned(n_strong=6, n_weak=6, n_neighbor=6):
    return {
        "strong": [{"text": f"strong wild sentence {i}", "act": 20.0, "rel": 0.5} for i in range(n_strong)],
        "weak": [{"text": f"weak wild sentence {i}", "act": 6.0, "rel": 0.15} for i in range(n_weak)],
        "neighbor": [{"text": f"neighbour sentence {i}", "of": 99} for i in range(n_neighbor)],
    }


def test_pool_adds_wild_and_neighbour_sentences(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    monkeypatch.setattr(backend_module, "load_wild", lambda layer, feature: scanned())
    pool = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2, n_wild=4, n_neighbor=3)
    by_source = {}
    for p in pool:
        by_source.setdefault(p["source"], []).append(p["text"])
    assert sorted(by_source["wild"]) == ["strong wild sentence 0", "strong wild sentence 1",
                                         "weak wild sentence 0", "weak wild sentence 1"]
    # Neighbour sentences are taken in order, whatever this feature does on them.
    assert sorted(by_source["neighbor"]) == ["neighbour sentence 0", "neighbour sentence 1", "neighbour sentence 2"]
    assert len(by_source["heldout"]) == 4 and len(by_source["random"]) == 2
    assert [p["id"] for p in pool] == list(range(13))
    # Same key, same pool; the default pool is unchanged by the new sources.
    assert pool == bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2, n_wild=4, n_neighbor=3)
    assert {p["source"] for p in bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)} == {"heldout", "random"}


def test_pool_refuses_when_too_few_wild_sentences_verify(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    monkeypatch.setattr(backend_module, "load_wild", lambda layer, feature: scanned(n_strong=2, n_weak=1))
    with pytest.raises(ValueError, match="only 3 .*wild"):
        bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2, n_wild=8, n_neighbor=2)
    monkeypatch.setattr(backend_module, "load_wild", lambda layer, feature: None)
    with pytest.raises(ValueError, match="scan_wild"):
        bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2, n_wild=8, n_neighbor=2)


def test_wild_sentences_are_checked_against_their_in_context_activation(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", load_45)
    monkeypatch.setattr(backend_module, "load_wild", lambda layer, feature: scanned())
    recorded_for = {}

    def fires(self, feature, layer, text, recorded=None):
        recorded_for[text] = recorded
        return "sentence 0" not in text  # the first candidate of each stratum loses its activation

    monkeypatch.setattr(Backend, "_window_fires", fires)
    pool = bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2, n_wild=4, n_neighbor=0)
    assert sorted(p["text"] for p in pool if p["source"] == "wild") == [
        "strong wild sentence 1", "strong wild sentence 2", "weak wild sentence 1", "weak wild sentence 2"]
    assert recorded_for["strong wild sentence 1"] == 20.0 and recorded_for["weak wild sentence 1"] == 6.0


# --- features whose examples peak on <bos> (DECISIONS 2026-11-15) -------------

def bos_examples(n_bos):
    def load(layer, feature):
        ex = distinct_examples(feature, 45 if feature == 7272 else 3)
        if feature == 7272:
            for e in ex[:n_bos]:
                e["tokens"][e["peak"]] = "<bos>"
        return ex
    return load


def test_pool_refuses_a_feature_whose_shown_examples_mostly_peak_on_bos(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", bos_examples(5))
    with pytest.raises(ValueError, match="5 of the 8 examples .*<bos>"):
        bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)


def test_pool_accepts_a_feature_with_half_its_shown_examples_on_bos(bare_backend, monkeypatch):
    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", bos_examples(4))
    assert len(bare_backend._pool(7272, 12, seed=0, n_heldout=4, n_random=2)) == 6


# --- review fixes, 2026-11-15 --------------------------------------------------

def test_two_features_do_not_share_an_id_to_source_layout(bare_backend, monkeypatch):
    def own_text(layer, feature):  # every feature has its own text, so nothing is dropped as a duplicate
        return [{"max": 1.0, "tokens": [f"▁f{feature}e{i}t{j}" for j in range(40)], "peak": 20} for i in range(45)]

    monkeypatch.setattr(backend_module.neuronpedia, "load_examples", own_text)
    layouts = {
        f: tuple(p["source"] for p in bare_backend._pool(f, 12, seed=1, n_heldout=6, n_random=10))
        for f in (7272, 7273, 7274)
    }
    assert all(sorted(v) == sorted(layouts[7272]) for v in layouts.values())  # same composition
    assert len(set(layouts.values())) == 3  # but the order is the feature's own
