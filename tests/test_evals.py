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


# --- blind-run fixes (FINDINGS 12) ---------------------------------------

def test_pick_random_features_depends_on_target_feature():
    from interp_playground.evals import pick_random_features

    hidden = [False] * 16384
    a = pick_random_features(16384, hidden, exclude=7272, seed=0, n=8)
    b = pick_random_features(16384, hidden, exclude=9435, seed=0, n=8)
    assert len(a) == len(b) == 8
    assert len(set(a) & set(b)) <= 1  # different features, different random pools
    assert a == pick_random_features(16384, hidden, exclude=7272, seed=0, n=8)


def test_spread_indices_covers_the_range():
    from interp_playground.evals import spread_indices

    idx = spread_indices(8, 45, 4)
    assert idx[0] == 8 and idx[-1] == 44 and len(idx) == 4
    assert idx == sorted(set(idx))
    assert spread_indices(8, 10, 4) == [8, 9]
    assert spread_indices(8, 8, 4) == []


def test_window_offsets_vary_and_sum_to_total():
    import random

    from interp_playground.evals import window_offsets

    rng = random.Random("t")
    draws = [window_offsets(rng) for _ in range(50)]
    assert all(b + a == 16 for b, a in draws)
    assert all(2 <= a <= 14 for _, a in draws)
    assert len({a for _, a in draws}) > 3


def test_paired_summary_counts_wins_and_sign_test():
    from interp_playground.evals import paired_summary

    def rec(f1_label, f1_inv, rho_label=0.5, rho_inv=0.6):
        return {"sets": {"label": {"f1": f1_label, "spearman": rho_label},
                         "investigated": {"f1": f1_inv, "spearman": rho_inv}}}

    recs = [rec(0.8, 0.9), rec(0.8, 0.9), rec(0.8, 0.9), rec(0.9, 0.9), rec(0.9, 0.8)]
    s = paired_summary(recs)
    assert s["n"] == 5
    assert s["f1"]["wins"] == 3 and s["f1"]["ties"] == 1 and s["f1"]["losses"] == 1
    assert s["f1"]["mean_delta"] == pytest.approx(0.04, abs=1e-6)
    # sign test on 3 wins / 1 loss, two-sided: 2 * (C(4,0)+C(4,1)) / 16 = 0.625
    assert s["f1"]["sign_p"] == pytest.approx(0.625)
    assert s["spearman"]["wins"] == 5 and s["spearman"]["sign_p"] == pytest.approx(2 / 32)
    assert paired_summary([{"sets": {"label": {"f1": None, "spearman": None}}}])["n"] == 0


# --- firing-checked pool (DECISIONS 2026-11-14) ----------------------------

def test_pick_firing_replaces_silent_picks_with_nearest_neighbours():
    from interp_playground.evals import pick_firing

    silent, calls = {8, 20, 21}, []

    def fires(i):
        calls.append(i)
        return i not in silent

    # Spread picks are 8, 20, 32, 44. 8 -> 9; 20 -> 21 is silent too -> 19.
    assert pick_firing(8, 45, 4, fires) == [9, 19, 32, 44]
    assert len(calls) == len(set(calls))  # no window is checked twice


def test_pick_firing_returns_fewer_when_nothing_else_fires():
    from interp_playground.evals import pick_firing

    assert pick_firing(8, 45, 4, lambda i: i in (30, 31)) == [30, 31]
    assert pick_firing(8, 12, 4, lambda i: False) == []
    assert pick_firing(8, 8, 4, lambda i: True) == []
    assert pick_firing(8, 45, 0, lambda i: True) == []


def test_value_errors_reach_the_client_with_their_message():
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    from interp_playground import server

    with pytest.raises(ToolError, match="16383") as exc:
        server.eval_sentences(feature=99999)
    assert not isinstance(exc.value, UnexpectedToolError)


# --- run-3 corrections (DECISIONS 2026-11-14, after run 3) -------------------

def test_paired_summary_compares_pool_rows_only():
    from interp_playground.evals import paired_summary

    def row(source, expected, rel):
        return {"source": source, "expected": expected, "rel": rel, "on": None, "peak_token": "x"}

    pool_label = [row("heldout", 0, 0.9), row("heldout", 9, 0.9), row("random", 0, 0.0)]
    pool_inv = [row("heldout", 0, 0.9), row("heldout", 9, 0.9), row("random", 0, 0.0)]
    agent = [row("agent", 9, 0.9), row("agent", 9, 0.8)]  # the agent's own sentences, all correct
    rec = {"sets": {"label": score_set([dict(r) for r in pool_label]),
                    "investigated": score_set([dict(r) for r in pool_inv + agent])}}
    assert rec["sets"]["investigated"]["f1"] > rec["sets"]["label"]["f1"]  # as recorded: inflated by agent rows
    s = paired_summary([rec])
    assert s["f1"]["ties"] == 1 and s["f1"]["wins"] == 0  # on the shared pool the two sets are identical
    assert s["f1"]["mean_delta"] == pytest.approx(0.0)


def test_shown_examples_are_not_held_out():
    from interp_playground.evals import duplicates_shown

    shown = ["We checked it out to find that the old bridge had been removed last year."]
    assert duplicates_shown("find that the  old bridge had been", shown)  # a window of a shown example
    assert not duplicates_shown("the new bridge had been", shown)
    assert not duplicates_shown("", shown)
