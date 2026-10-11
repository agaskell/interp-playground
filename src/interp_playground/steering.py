"""Pure helpers for steering experiments: control directions and effect metrics."""

import torch


def random_direction(d_model: int, seed: int) -> torch.Tensor:
    """A seeded random unit vector: the control for a feature direction.

    In high dimensions it's nearly orthogonal to everything the model uses,
    so it separates "this feature does X" from "pushing this hard does X".
    """
    gen = torch.Generator().manual_seed(seed)
    v = torch.randn(d_model, generator=gen)
    return v / v.norm()


def steer_metrics(
    base_probs: torch.Tensor,
    probs: torch.Tensor,
    promoted: list[int],
    completion_ids: list[int],
) -> dict[str, float | int]:
    """Effect of one steered run, from next-token distributions at the prompt's end.

    - p_promoted: probability mass on the promoted tokens.
    - kl: KL(baseline || steered), how far the whole distribution moved.
    - promoted_in_completion: generated tokens that are promoted tokens.
    """
    eps = 1e-12
    kl = (base_probs * ((base_probs + eps) / (probs + eps)).log()).sum().item()
    promoted_set = set(promoted)
    return {
        "p_promoted": round(probs[promoted].sum().item(), 4),
        "kl": round(max(kl, 0.0), 4),
        "promoted_in_completion": sum(t in promoted_set for t in completion_ids),
    }
