import math

import pytest
import torch

from interp_playground.steering import random_direction, steer_metrics


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
