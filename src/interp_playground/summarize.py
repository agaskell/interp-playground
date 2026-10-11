"""Turn raw SAE activations into compact, agent-friendly summaries.

Pure functions over tensors, separate from the model so they're testable
without loading it.
"""

from typing import Any

import torch

from interp_playground.neuronpedia import FeatureMeta

TOP_FEATURES = 10
# A feature is a position artifact if it fires at the first position of at
# least this fraction of probe prompts and never fires anywhere else in them.
ARTIFACT_MIN_FRACTION = 0.75


def find_position_artifacts(probe_acts: list[torch.Tensor]) -> torch.Tensor:
    """Mask of features that respond to "first token" rather than to content.

    Each tensor is [seq, d_sae] for one probe prompt, <bos> already dropped.
    The first token sees nothing before it, so its residual stream is
    unusual and the SAE dedicates features to reconstructing it.
    """
    first = torch.stack([a[0] > 0 for a in probe_acts]).float().mean(0)
    later = torch.stack([(a[1:] > 0).any(0) for a in probe_acts]).any(0)
    return (first >= ARTIFACT_MIN_FRACTION) & ~later


def summarize(
    acts: torch.Tensor,
    tokens: list[str],
    max_act: torch.Tensor,
    hidden: dict[str, torch.Tensor],
    table: dict[int, FeatureMeta],
    top_k: int,
    watch: list[int],
    include_hidden: bool,
) -> dict[str, Any]:
    """Summarize [seq, d_sae] activations for `tokens` (no <bos>).

    Positions in the output are 0-based indexes into `tokens`.
    """
    d_sae = acts.shape[1]
    for feature in watch:
        if not 0 <= feature < d_sae:
            raise ValueError(f"feature {feature} out of range 0-{d_sae - 1}")

    raw = acts
    if not include_hidden:
        for mask in hidden.values():
            acts = acts.masked_fill(mask, 0.0)
    # Rank by activation relative to the feature's own max, so a strong
    # firing of a quiet feature beats a middling firing of a loud one.
    rel = acts / max_act.clamp(min=1e-6)

    def entry(feature: int, pos: int) -> dict[str, Any]:
        return {
            "feature": feature,
            "act": round(acts[pos, feature].item(), 2),
            "rel": round(rel[pos, feature].item(), 2),
        }

    # Whole-text view: each feature at its strongest position.
    best_rel, best_pos = rel.max(dim=0)
    n = min(TOP_FEATURES, int((best_rel > 0).sum()))
    top_features = []
    for feature in best_rel.topk(n).indices.tolist():
        pos = best_pos[feature].item()
        meta = table[feature]
        top_features.append(
            {
                **entry(feature, pos),
                "at": pos,
                "token": tokens[pos],
                "label": meta.label,
                "density": round(meta.density, 5),
                "promotes": meta.promotes,
            }
        )

    out: dict[str, Any] = {"tokens": tokens, "top_features": top_features}

    if top_k > 0:
        out["per_token"] = []
        for pos in range(len(tokens)):
            top = rel[pos].topk(min(top_k, d_sae))
            out["per_token"].append(
                [entry(f, pos) for v, f in zip(top.values.tolist(), top.indices.tolist()) if v > 0]
            )

    if watch:
        out["watched"] = {}
        for feature in watch:
            meta = table[feature]
            reason = next((name for name, mask in hidden.items() if mask[feature]), None)
            out["watched"][str(feature)] = {
                "label": meta.label,
                "density": round(meta.density, 5),
                "max_act": meta.max_act,
                "hidden": reason,
                "acts": [round(a, 2) for a in raw[:, feature].tolist()],
            }

    out["hidden"] = {
        name: 0 if include_hidden else int(mask.sum()) for name, mask in hidden.items()
    }
    return out
