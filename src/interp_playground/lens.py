"""Pure helpers for the logit lens."""

from collections.abc import Callable
from typing import Any

import torch


def lens_rows(
    probs: torch.Tensor, layers: list[int], decode: Callable[[int], str], top_k: int
) -> list[dict[str, Any]]:
    """Per-layer top tokens from [n_layers, vocab] next-token distributions.

    `p_final` is each layer's probability for the token the last layer
    predicts: how early the final answer is already present.
    """
    final = int(probs[-1].argmax())
    rows = []
    for layer, p in zip(layers, probs):
        top = p.topk(top_k)
        rows.append(
            {
                "layer": layer,
                "top": [[decode(i), round(v, 4)] for v, i in zip(top.values.tolist(), top.indices.tolist())],
                "p_final": round(p[final].item(), 4),
            }
        )
    return rows
