"""Steer generation by adding an SAE feature's decoder direction to the residual stream.

Usage: uv run python scripts/steer.py [--feature N] [--layer L] [--scales S ...] [prompt]

Pass --random SEED to steer along a random unit direction instead: a control
for whether an effect is specific to the feature or just from pushing hard.
"""

import argparse

import torch

from interp_playground.loading import load_model, load_sae


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature", type=int, default=7272)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--scales", type=float, nargs="+", default=[0, 50, 100, 200, 400])
    parser.add_argument("--max-new-tokens", type=int, default=40)
    parser.add_argument("--random", type=int, metavar="SEED",
                        help="steer along a random unit direction (control)")
    parser.add_argument("prompt", nargs="?", default="My favorite thing to do on weekends is")
    args = parser.parse_args()

    model = load_model()
    sae = load_sae(args.layer)
    layers = model.model.layers
    # Decoder rows are unit-norm, so `scale` is in the feature's activation units.
    direction = sae.W_dec[args.feature].detach()
    label = f"feature {args.feature}"
    if args.random is not None:
        gen = torch.Generator().manual_seed(args.random)
        rand = torch.randn(direction.shape, generator=gen)
        direction = (rand / rand.norm()).to(direction.device)
        label = f"random direction (seed {args.random})"

    with torch.no_grad(), model.trace(args.prompt):
        resid = layers[args.layer].output.save()
    # Skip <bos>, whose residual norm is an outlier.
    resid_norm = resid[0, 1:].norm(dim=-1).mean().item()
    print(f"{label} @ layer {args.layer}; "
          f"typical residual norm here: {resid_norm:.0f}\n")

    prompt_len = len(model.tokenizer(args.prompt)["input_ids"])
    for scale in args.scales:
        with torch.no_grad(), model.generate(
            args.prompt, max_new_tokens=args.max_new_tokens, do_sample=False
        ) as tracer:
            # Every forward pass: the prompt, then each generated token.
            with tracer.all():
                layer = layers[args.layer]
                layer.output = layer.output + scale * direction
            tokens = tracer.result.save()

        completion = model.tokenizer.decode(tokens[0, prompt_len:])
        print(f"--- scale {scale:g} ({scale / resid_norm:.0%} of residual norm)")
        print(f"{args.prompt}\033[1m{completion}\033[0m\n")


if __name__ == "__main__":
    main()
