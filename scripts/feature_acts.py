"""Show where one SAE feature fires, token by token.

Usage: uv run python scripts/feature_acts.py [--feature N] [--layer L] [prompt ...]
"""

import argparse

import torch

from interp_playground.loading import load_model, load_sae

DEFAULT_PROMPTS = [
    "We drove across the old bridge into town.",
    "The dentist fitted a bridge to replace my missing tooth.",
    "In the card game bridge, each player bids in turn.",
    "The Golden Gate is the most photographed landmark in San Francisco.",
    "The ferry carried us across the river to the other side.",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=int, default=7272)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("prompts", nargs="*", default=DEFAULT_PROMPTS)
    args = parser.parse_args()

    model = load_model()
    sae = load_sae(args.layer)
    tokenizer = model.tokenizer

    for prompt in args.prompts:
        with torch.no_grad(), model.trace(prompt):
            resid = model.model.layers[args.layer].output.save()

        with torch.no_grad():
            acts = sae.encode(resid[0])  # [seq, d_sae]

        # Position 0 is <bos>. Its residual norm is huge and Gemma Scope was
        # trained without it, so skip it.
        ids = tokenizer(prompt)["input_ids"]
        feat = acts[1:, args.feature].tolist()
        l0 = (acts[1:] > 0).sum(-1).float().mean().item()

        print(f"\n{prompt!r}   (avg features active per token: {l0:.0f})")
        for tok_id, a in zip(ids[1:], feat):
            bar = "#" * round(a / 2)
            print(f"  {tokenizer.decode(tok_id)!r:>14}  {a:6.2f}  {bar}")

        peak = int(acts[1:, args.feature].argmax()) + 1
        if acts[peak, args.feature] > 0:
            top = acts[peak].topk(5)
            others = ", ".join(
                f"{i}={v:.1f}" for v, i in zip(top.values.tolist(), top.indices.tolist())
            )
            print(f"  top-5 features at {tokenizer.decode(ids[peak])!r}: {others}")


if __name__ == "__main__":
    main()
