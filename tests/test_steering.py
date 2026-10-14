import math

import pytest
import torch

from interp_playground.steering import (
    completion_metrics,
    position_mask,
    random_direction,
    steer_metrics,
    teacher_mask,
)


def test_metrics_for_unchanged_distribution():
    base = torch.tensor([0.5, 0.25, 0.25])
    m = steer_metrics(base, base, promoted=[1], completion_ids=[0, 1, 1, 2])
    assert m == {"p_promoted": 0.25, "kl": 0.0, "promoted_in_completion": 2}


def test_kl_is_from_baseline_to_steered():
    base = torch.tensor([0.5, 0.5])
    steered = torch.tensor([0.9, 0.1])
    m = steer_metrics(base, steered, promoted=[0], completion_ids=[])
    expected = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)
    assert m["kl"] == pytest.approx(expected, abs=1e-3)
    assert m["p_promoted"] == pytest.approx(0.9)


def test_random_direction_is_unit_norm_and_seeded():
    a = random_direction(16, seed=3)
    assert a.norm().item() == pytest.approx(1.0)
    assert torch.equal(a, random_direction(16, seed=3))
    assert not torch.equal(a, random_direction(16, seed=4))


def test_target_tokens_are_reported_separately():
    base = torch.tensor([0.5, 0.25, 0.25])
    m = steer_metrics(base, base, promoted=[1], completion_ids=[2, 2], target=[2])
    assert m["p_target"] == 0.25 and m["target_in_completion"] == 2


def test_position_mask():
    assert position_mask(4, "all", True) is None
    assert position_mask(4, "last", False).squeeze(1).tolist() == [0, 0, 0, 1]
    assert position_mask(4, "prompt", True) is None
    assert position_mask(1, "prompt", False).tolist() == [[0.0]]
    with pytest.raises(ValueError):
        position_mask(4, "first", True)


def test_teacher_mask_matches_generation_time_edits():
    # prompt of 3 tokens, 2 generated tokens: positions 0-2 prompt, 3-4 completion
    assert teacher_mask(5, 3, "all") is None
    assert teacher_mask(5, 3, "last").squeeze(1).tolist() == [0, 0, 1, 1, 1]
    assert teacher_mask(5, 3, "prompt").squeeze(1).tolist() == [1, 1, 1, 0, 0]
    with pytest.raises(ValueError):
        teacher_mask(5, 3, "first")


def test_completion_metrics_average_over_steps():
    steps = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]])
    m = completion_metrics(steps, completion_ids=[0, 1], promoted=[2], target=[1])
    assert m == {"p_promoted_completion": 0.2, "p_target_completion": 0.4}
    m = completion_metrics(steps, completion_ids=[0, 0], promoted=[2])
    assert m["teacher_forced_mismatch"] == 1
