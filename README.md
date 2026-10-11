# interp-playground

Tools for looking inside a language model: activations, sparse-autoencoder (SAE)
features, steering, and ablation. The goal is an MCP server that lets an AI agent
and a human investigate a model's internals together.

The motivation: humans *and* AIs should be able to understand how a model
reasons, especially if chain-of-thought stops being legible English. This makes
that concrete: an AI using tools to understand another model.

It's a learning project. Current target: **Gemma 2 2B** with Google DeepMind's
**Gemma Scope** residual-stream SAEs, running locally on Apple Silicon (MPS).

## Status

- [x] Nix + uv environment; MPS verified against CPU
- [x] Reproduce a known SAE feature by hand (activations, steering, random control)
- [x] MCP server: `run_prompt`, `feature_info`, `steer`, `ablate`, `logit_lens`
- [x] First agent-driven investigations (see [FINDINGS.md](FINDINGS.md))

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

## MCP server

```sh
claude mcp add interp-playground -- uv run --directory /path/to/interp-playground interp-playground-mcp
uv run python scripts/mcp_smoke.py   # end-to-end check over stdio
```

`run_prompt` returns the top SAE features across the text and per token, with
Neuronpedia labels, plus exact per-token activations for any features you ask
about (`features=[...]`). Activations are also given relative to each feature's
typical max (`rel`). Two kinds of features are hidden from rankings by default:
dense ones (active on >10% of tokens) and position artifacts (detected at load
time with probe prompts). The model stays loaded between calls. Feature metadata
comes from Neuronpedia's bulk exports and is cached in `~/.cache/interp-playground/`.

`steer` adds a feature's direction during generation at several scales (default
1x/2x/4x the feature's max activation), each next to a random-direction control
of the same size. It reports the completions plus `p_promoted` (next-token
probability on the feature's promoted tokens) and `kl` (how far the distribution
moved), so feature-specific effects can be separated from generic disruption.

`ablate` removes a feature's own contribution (or clamps it) at the positions
where it fires and reports the change in next-token predictions, per-position
log-prob deltas, and optionally completions. `feature_info` gathers a feature's
label, density, promoted/suppressed tokens, nearest features by decoder
direction, and Neuronpedia's top examples. `logit_lens` shows what each layer
would predict at a position, and when the final answer forms.

```sh
uv run pytest            # unit tests (no model needed)
```

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

The full log is in [FINDINGS.md](FINDINGS.md). Highlights:

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

**Position artifacts.** Features 1178, 4667, 6889 and 11087 (layer 12) fire on
whatever token comes first, whether "Bananas", "def" or "1", and nowhere else.
Their density is ~1/127, i.e. once per Neuronpedia text window. Their
auto-generated labels ("words that indicate agreement, concession...") are
confabulated meanings for a purely positional signal. Found by Claude using
`run_prompt`; this is why the server now detects and hides them.

## Roadmap

The initial tool set is complete. Design principles:

- **Compact, structured JSON.** Agents do badly with large activation dumps;
  return top-k lists, deltas and exact values for features asked about.
- **Swappable backend.** Keep model access behind an interface so larger models,
  or remote execution via [NDIF](https://nnsight.net), can be added later.
- **Long-lived process.** Load the model once, not per call.

Future directions:

- **Larger models:** Gemma 2 9B locally; much larger via NDIF.
- **Probes on a reasoning model's chain of thought:** does the model "know" its
  answer before its written reasoning says so? A CoT-faithfulness question.
- **Train our own SAEs** where pretrained ones don't cover a model or layer
  (activation caching, storage, streaming, shuffling).
- **Evaluate the agent's explanations:** do its claims about features hold up
  under intervention? An open question in automated interpretability.

References: [SAELens](https://github.com/jbloomAus/SAELens),
[nnsight](https://nnsight.net), [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens),
[Neuronpedia](https://www.neuronpedia.org), [Gemma Scope](https://huggingface.co/google/gemma-scope),
and [ARENA](https://arena.education)'s mechanistic interpretability chapter.

## Notes

- **fp32, not bf16.** bf16 on MPS drifts 1-4% in the residual stream, enough to
  flip JumpReLU features near their thresholds.
- **nnsight 0.7:** only objects that are themselves `.save()`d leave a trace.
  Wrap lists of saved tensors in `nnsight.save([...])`.
- `transformer-lens` is pinned below 4.0 until sae-lens supports it.

## License

MIT
