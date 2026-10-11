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
    target: list[int] | None = None,
) -> dict[str, float | int]:
    """Effect of one steered run, from next-token distributions at the prompt's end.

    - p_promoted: probability mass on the promoted tokens.
    - kl: KL(baseline || steered), how far the whole distribution moved.
    - promoted_in_completion: generated tokens that are promoted tokens.
    - p_target / target_in_completion: the same for caller-chosen tokens, if given.
    """
    eps = 1e-12
    kl = (base_probs * ((base_probs + eps) / (probs + eps)).log()).sum().item()
    out: dict[str, float | int] = {
        "p_promoted": round(probs[promoted].sum().item(), 4),
        "kl": round(max(kl, 0.0), 4),
        "promoted_in_completion": sum(t in set(promoted) for t in completion_ids),
    }
    if target:
        out["p_target"] = round(probs[target].sum().item(), 4)
        out["target_in_completion"] = sum(t in set(target) for t in completion_ids)
    return out


def position_mask(seq_len: int, positions: str, is_prompt_pass: bool) -> torch.Tensor | None:
    """[seq_len, 1] mask of positions to steer, or None for all of them.

    - "all": every position on every forward pass.
    - "last": the final position only; during generation that's the token
      being produced at each step.
    - "prompt": every position of the prompt pass, nothing on later passes.
    """
    if positions == "all":
        return None
    if positions == "last":
        mask = torch.zeros(seq_len, 1)
        mask[-1] = 1.0
        return mask
    if positions == "prompt":
        return None if is_prompt_pass else torch.zeros(seq_len, 1)
    raise ValueError(f"positions must be all, last or prompt, not {positions!r}")
