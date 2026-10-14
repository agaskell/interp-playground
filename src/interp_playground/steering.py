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


def teacher_mask(seq_len: int, prompt_len: int, positions: str) -> torch.Tensor | None:
    """[seq_len, 1] mask that reproduces a generation-time edit in one pass.

    Greedy generation under `position_mask` leaves the KV cache holding edited
    states at some positions and clean ones at others. Re-running prompt +
    completion in a single teacher-forced pass with this mask gives the same
    states, so per-step distributions can be read off without re-generating.
    """
    if positions == "all":
        return None
    mask = torch.zeros(seq_len, 1)
    if positions == "last":
        mask[prompt_len - 1:] = 1.0  # last prompt token, then each generated token
    elif positions == "prompt":
        mask[:prompt_len] = 1.0
    else:
        raise ValueError(f"positions must be all, last or prompt, not {positions!r}")
    return mask


def completion_metrics(
    step_probs: torch.Tensor,
    completion_ids: list[int],
    promoted: list[int],
    target: list[int] | None = None,
) -> dict[str, float | int]:
    """Metrics averaged over the completion, from per-step next-token distributions.

    `step_probs` is [steps, vocab]: the distribution that produced each
    completion token. Mean mass on a token set catches "the model kept wanting
    to say funeral" even where greedy picked something else, which a single
    read at the prompt's end misses entirely.

    - p_promoted_completion / p_target_completion: mean mass per step.
    - teacher_forced_mismatch: steps whose argmax isn't the generated token.
      Should be 0; anything else means the re-run didn't match generation.
    """
    out: dict[str, float | int] = {
        "p_promoted_completion": round(step_probs[:, promoted].sum(-1).mean().item(), 4),
    }
    if target:
        out["p_target_completion"] = round(step_probs[:, target].sum(-1).mean().item(), 4)
    mismatch = int((step_probs.argmax(-1) != torch.tensor(completion_ids)).sum().item())
    if mismatch:
        out["teacher_forced_mismatch"] = mismatch
    return out
