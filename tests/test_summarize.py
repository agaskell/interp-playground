import pytest
import torch

from interp_playground.neuronpedia import FeatureMeta
from interp_playground.summarize import find_position_artifacts, summarize

TOKENS = ["a", "b", "c"]
# 3 tokens x 4 features. Feature 0 is "dense", feature 3 a position artifact.
ACTS = torch.tensor(
    [
        [5.0, 1.0, 0.0, 40.0],
        [5.0, 8.0, 0.0, 0.0],
        [5.0, 0.0, 3.0, 0.0],
    ]
)
MAX_ACT = torch.tensor([10.0, 10.0, 6.0, 50.0])
HIDDEN = {
    "dense": torch.tensor([True, False, False, False]),
    "position_artifact": torch.tensor([False, False, False, True]),
}
TABLE = {i: FeatureMeta(label=f"f{i}", density=0.01, max_act=float(MAX_ACT[i]), promotes=[])
         for i in range(4)}


def run(**kw):
    args = dict(acts=ACTS, tokens=TOKENS, max_act=MAX_ACT, hidden=HIDDEN, table=TABLE,
                top_k=2, watch=[], include_hidden=False)
    return summarize(**(args | kw))


def test_hidden_features_excluded_from_rankings():
    out = run()
    ranked = {f["feature"] for f in out["top_features"]}
    assert ranked == {1, 2}
    assert all(f["feature"] in (1, 2) for tok in out["per_token"] for f in tok)
    assert out["hidden"] == {"dense": 1, "position_artifact": 1}


def test_include_hidden_restores_them():
    out = run(include_hidden=True)
    assert {f["feature"] for f in out["top_features"]} == {0, 1, 2, 3}
    assert out["hidden"] == {"dense": 0, "position_artifact": 0}


def test_top_features_ranked_by_rel_at_peak_token():
    first = run()["top_features"][0]
    assert first["feature"] == 1  # rel 0.8 beats feature 2's rel 0.5
    assert (first["at"], first["token"], first["act"], first["rel"]) == (1, "b", 8.0, 0.8)


def test_watched_features_report_every_token_even_if_hidden():
    out = run(watch=[3, 2])
    assert out["watched"]["3"]["acts"] == [40.0, 0.0, 0.0]
    assert out["watched"]["3"]["hidden"] == "position_artifact"
    assert out["watched"]["2"]["acts"] == [0.0, 0.0, 3.0]
    assert out["watched"]["2"]["hidden"] is None


def test_top_k_zero_omits_per_token_lists():
    out = run(top_k=0, watch=[1])
    assert "per_token" not in out
    assert out["tokens"] == TOKENS


def test_watch_rejects_out_of_range_feature():
    with pytest.raises(ValueError):
        run(watch=[4])


def test_find_position_artifacts():
    # Feature 0: first position in 3/4 probes, never later -> artifact.
    # Feature 1: first position in 4/4 probes but also later -> not an artifact.
    # Feature 2: first position in 2/4 probes -> not an artifact.
    probes = [
        torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 1.0], [0.0, 2.0, 0.0]]),
        torch.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
        torch.tensor([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    ]
    assert find_position_artifacts(probes).tolist() == [True, False, False]
