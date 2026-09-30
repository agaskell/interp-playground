# interp-playground

Tools for looking inside a language model: activations, sparse-autoencoder (SAE)
features, steering, and ablation. The goal is an MCP server that lets an AI agent
and a human investigate a model's internals together.

It's a learning project. Current target: **Gemma 2 2B** with Google DeepMind's
**Gemma Scope** residual-stream SAEs, running locally on Apple Silicon (MPS).

## Status

- [x] Nix + uv environment; MPS verified against CPU
- [x] Reproduce a known SAE feature by hand (activations, steering, random control)
- [ ] MCP server: `run_prompt`, `feature_info`, `steer`, `ablate`, `logit_lens`

## Setup

Requires Nix with flakes. The flake provides Python 3.12 and uv; uv manages
Python packages.

```sh
nix develop            # or: direnv allow
uv sync
```

Gemma 2 is gated on Hugging Face. Accept the license at
[google/gemma-2-2b](https://huggingface.co/google/gemma-2-2b), then log in once:

```sh
uv run hf auth login
```

The model (~10 GB in fp32) and SAEs download on first use.

## Scripts

```sh
# Does MPS give the same internals as CPU? (fp32: yes; bf16: 1-4% drift)
uv run python scripts/check_mps.py

# Where does a feature fire, token by token?
uv run python scripts/feature_acts.py --feature 7272 "We drove across the old bridge."

# Steer generation along a feature direction, or a random one as a control
uv run python scripts/steer.py --feature 7272 --scales 0 50 100 200
uv run python scripts/steer.py --random 0 --scales 0 50 100 200
```

## Findings so far

Layer-12 feature **7272**, labelled "references to various types of bridges" on
[Neuronpedia](https://www.neuronpedia.org/gemma-2-2b/12-gemmascope-res-16k/7272):

- **Activations:** fires at 35-50 on the word "bridge" in every sense
  (physical, dental, card game, metaphorical) and at 4-22 on bridge-like concepts
  without the word (overpass, ferry, river). The label describes the top of the
  activation range; the low range is broader.
- **Steering:** at scale 70-90 the output drifts toward rivers and "connecting
  the community"; at 100+ it produces "bridge" and degrades. Random directions of
  the same norm never produce "bridge" and stay fluent up to ~200, so the effect
  is specific to the feature. (One prompt, two random seeds: suggestive, not
  conclusive.)

## Notes

- **fp32, not bf16.** bf16 on MPS drifts 1-4% in the residual stream, enough to
  flip JumpReLU features near their thresholds.
- **nnsight 0.7:** only objects that are themselves `.save()`d leave a trace.
  Wrap lists of saved tensors in `nnsight.save([...])`.
- `transformer-lens` is pinned below 4.0 until sae-lens supports it.

## License

MIT
