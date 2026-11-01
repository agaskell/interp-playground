import pytest

from interp_playground.evals import (
    FIRE_REL,
    match_predictions,
    peak_matches,
    score_set,
    spearman,
)

POOL = [
    {"id": 0, "text": "We drove across the old bridge.", "source": "heldout"},
    {"id": 1, "text": "The soup needs more salt.", "source": "random"},
    {"id": 2, "text": "Music can bridge divides.", "source": "heldout"},
]


def test_spearman_ranks_and_clamps_weak_activations():
    # Perfect ranking once sub-threshold rels are treated as 0.
    assert spearman([0, 5, 10], [0.0, 0.5, 1.0]) == 1.0
    # Ranking among non-firing sentences must not count: 0.02 vs 0.05 are both 0.
    assert spearman([0, 1, 10], [0.05, 0.02, 1.0]) == spearman([1, 0, 10], [0.05, 0.02, 1.0])


def test_spearman_none_on_constant_or_empty():
    assert spearman([0, 0, 0], [0.0, 0.5, 1.0]) is None
    assert spearman([], []) is None
    assert spearman([3], [0.5]) is None


def test_peak_matches_is_lenient_about_spacing_case_and_plurals():
    assert peak_matches("bridge", " bridges")
    assert peak_matches(" Bridges", "bridge")
    assert peak_matches("San Francisco", " San")
    assert not peak_matches("river", " bridge")
    assert not peak_matches(None, " bridge")
    assert not peak_matches("", " bridge")


def test_match_predictions_by_id_and_agent_text():
    preds = [
        {"id": 0, "expected": 9, "on": "bridge"},
        {"id": 1, "expected": 0},
        {"id": 2, "expected": 4, "on": "bridge"},
        {"text": "The dentist fitted a bridge.", "expected": 7, "on": "bridge"},
    ]
    rows = match_predictions(POOL, preds)
    assert [r["source"] for r in rows] == ["heldout", "random", "heldout", "agent"]
    assert rows[3]["id"] is None and rows[3]["text"] == "The dentist fitted a bridge."
    assert rows[1]["on"] is None


def test_match_predictions_requires_every_pool_id():
    with pytest.raises(ValueError, match=r"missing.*\[1, 2\]"):
        match_predictions(POOL, [{"id": 0, "expected": 9}])


def test_match_predictions_rejects_duplicates_and_blanks():
    with pytest.raises(ValueError, match="twice"):
        match_predictions(POOL, [{"id": 0, "expected": 9}, {"id": 0, "expected": 1},
                                 {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    with pytest.raises(ValueError, match="id .* or text"):
        match_predictions(POOL, [{"expected": 9}, {"id": 0, "expected": 9},
                                 {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    with pytest.raises(ValueError, match="unknown sentence id 7"):
        match_predictions(POOL, [{"id": 7, "expected": 9}, {"id": 0, "expected": 9},
                                 {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])


def test_match_predictions_coerces_expected():
    rows = match_predictions(POOL, [{"id": 0, "expected": "7"}, {"id": 1, "expected": -3},
                                    {"id": 2, "expected": 15}])
    assert [r["expected"] for r in rows] == [7.0, 0.0, 10.0]
    with pytest.raises(ValueError, match="expected"):
        match_predictions(POOL, [{"id": 0, "expected": "high"}, {"id": 1, "expected": 0},
                                 {"id": 2, "expected": 0}])


def test_agent_text_matching_pool_maps_to_pool_id():
    rows = match_predictions(POOL, [{"text": "We drove across the old bridge.", "expected": 9},
                                    {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    assert rows[0]["id"] == 0 and rows[0]["source"] == "heldout"


def test_match_predictions_caps_agent_sentences():
    extra = [{"text": f"agent sentence {i}", "expected": 0} for i in range(5)]
    with pytest.raises(ValueError, match="at most 4 agent"):
        match_predictions(POOL, [{"id": 0, "expected": 0}, {"id": 1, "expected": 0},
                                 {"id": 2, "expected": 0}, *extra])


def test_score_set_metrics():
    rows = [
        # fired, predicted, peak hit
        {"id": 0, "text": "a", "source": "heldout", "expected": 9.0, "on": "bridge", "rel": 1.0, "peak_token": " bridge"},
        # fired, predicted, peak miss
        {"id": 2, "text": "b", "source": "heldout", "expected": 4.0, "on": "music", "rel": 0.3, "peak_token": " bridge"},
        # not fired, not predicted
        {"id": 1, "text": "c", "source": "random", "expected": 0.0, "on": None, "rel": 0.0, "peak_token": " salt"},
        # not fired (weak), predicted: false positive
        {"id": None, "text": "d", "source": "agent", "expected": 5.0, "on": "gap", "rel": 0.05, "peak_token": " gap"},
        # fired, not predicted: false negative, no `on` so excluded from peak_hit
        {"id": None, "text": "e", "source": "agent", "expected": 1.0, "on": None, "rel": 0.2, "peak_token": " span"},
    ]
    s = score_set(rows)
    assert s["n"] == 5
    assert (s["precision"], s["recall"]) == (pytest.approx(2 / 3, abs=1e-3), pytest.approx(2 / 3, abs=1e-3))
    assert s["f1"] == pytest.approx(2 / 3, abs=1e-3)
    assert s["peak_hit"] == 0.5
    assert s["by_source"]["heldout"] == {"n": 2, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert s["by_source"]["random"] == {"n": 1, "precision": None, "recall": None, "f1": None}
    assert s["by_source"]["agent"]["precision"] == 0.0 and s["by_source"]["agent"]["recall"] == 0.0
    assert s["by_source"]["agent"]["f1"] == 0.0
    assert rows[3]["fired"] is False and rows[3]["predicted_fired"] is True
    assert rows[3]["peak_hit"] is None  # not fired, so no peak to hit
    assert rows[0]["peak_hit"] is True and rows[1]["peak_hit"] is False
    assert s["spearman"] is not None and FIRE_REL == 0.1


from interp_playground.evals import build_pool, pick_random_features


def test_pick_random_features_skips_hidden_and_self_and_is_seeded():
    hidden = [False] * 20
    hidden[3] = hidden[4] = True
    picked = pick_random_features(20, hidden, exclude=7, seed=0, n=10)
    assert len(picked) == len(set(picked)) == 10
    assert not {3, 4, 7} & set(picked)
    assert picked == pick_random_features(20, hidden, exclude=7, seed=0, n=10)
    assert picked != pick_random_features(20, hidden, exclude=7, seed=1, n=10)
    # Asking for more than exist returns what exists.
    assert len(pick_random_features(5, [False] * 5, exclude=0, seed=0, n=10)) == 4


def test_build_pool_shuffles_deterministically_and_tags_sources():
    pool = build_pool(["h1", "h2"], ["r1", "r2", "r3"], seed=0)
    assert [p["id"] for p in pool] == [0, 1, 2, 3, 4]
    assert sorted(p["text"] for p in pool) == ["h1", "h2", "r1", "r2", "r3"]
    assert {p["text"]: p["source"] for p in pool} == {
        "h1": "heldout", "h2": "heldout", "r1": "random", "r2": "random", "r3": "random"
    }
    assert pool == build_pool(["h1", "h2"], ["r1", "r2", "r3"], seed=0)
    assert [p["text"] for p in pool] != [p["text"] for p in build_pool(["h1", "h2"], ["r1", "r2", "r3"], seed=1)]


def test_build_pool_with_empty_sources():
    assert build_pool([], [], seed=0) == []
    pool = build_pool([], ["r1"], seed=0)
    assert pool == [{"id": 0, "text": "r1", "source": "random"}]
    # Grading an empty pool with only agent sentences still scores.
    rows = match_predictions([], [{"text": "x", "expected": 3}])
    rows[0].update({"rel": 0.0, "peak_token": " x"})
    s = score_set(rows)
    # One false positive: precision 0, recall undefined (nothing fired), so f1 is None.
    assert s["n"] == 1 and s["spearman"] is None
    assert s["precision"] == 0.0 and s["recall"] is None and s["f1"] is None


# --- review fixes ---------------------------------------------------------

from interp_playground.evals import relative_activation


def test_match_predictions_requires_expected_and_rejects_non_finite():
    with pytest.raises(ValueError, match="no 'expected'"):
        match_predictions(POOL, [{"id": 0}, {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    with pytest.raises(ValueError, match="no 'expected'"):
        match_predictions(POOL, [{"id": 0, "activation": 9}, {"id": 1, "expected": 0},
                                 {"id": 2, "expected": 0}])
    with pytest.raises(ValueError, match="expected"):
        match_predictions(POOL, [{"id": 0, "expected": "nan"}, {"id": 1, "expected": 0},
                                 {"id": 2, "expected": 0}])


def test_match_predictions_rejects_non_string_on():
    with pytest.raises(ValueError, match="on must be a string"):
        match_predictions(POOL, [{"id": 0, "expected": 9, "on": 1987}, {"id": 1, "expected": 0},
                                 {"id": 2, "expected": 0}])


def test_match_predictions_cross_checks_text_against_id():
    # Same id, different sentence: the pool the agent predicted on is not this pool.
    with pytest.raises(ValueError, match=r"id 0 is .*old bridge.* in this pool"):
        match_predictions(POOL, [{"id": 0, "text": "The soup needs more salt.", "expected": 9},
                                 {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    # Whitespace differences are not a mismatch, with or without an id.
    rows = match_predictions(POOL, [{"id": 0, "text": "We drove  across the old bridge.", "expected": 9},
                                    {"text": "Music can  bridge divides.", "expected": 4},
                                    {"id": 1, "expected": 0}])
    assert [r["id"] for r in rows] == [0, 2, 1]


def test_build_pool_drops_duplicate_texts_keeping_heldout():
    pool = build_pool(["same", "h2"], ["same", "r2", "r2"], seed=0)
    texts = [p["text"] for p in pool]
    assert sorted(texts) == ["h2", "r2", "same"]
    assert next(p for p in pool if p["text"] == "same")["source"] == "heldout"
    assert [p["id"] for p in pool] == [0, 1, 2]


def test_relative_activation_guards_zero_max():
    assert relative_activation(5.0, 50.0) == 0.1
    assert relative_activation(0.0, 0.0) == 0.0
    assert relative_activation(3.0, 0.0) > 1.0  # dead-feature table entry: no ZeroDivisionError
