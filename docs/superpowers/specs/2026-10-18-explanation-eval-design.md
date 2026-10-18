# Agent explanation eval: design

Date: 2026-10-18. Status: approved in conversation; self-reviewed; pending Andy's review.

## Purpose

Test whether an agent's explanation of an SAE feature predicts where the
feature fires. Sections 3 and 7 of FINDINGS.md did this by hand (position
artifacts with confabulated labels; 9435 "San Francisco" being a "San"
string feature). This builds the automated version on the existing tools,
in the spirit of simulation scoring from automated interpretability
(Bills et al. 2023; Neuronpedia's auto-interp scoring).

The deliverable is a comparison, per feature, of two explanations:

- **label**: predictions made from the Neuronpedia label alone.
- **investigated**: predictions made after the agent studied the feature
  with `feature_info`, `run_prompt`, `steer` and so on.

It answers "does investigating help, or is the label as good?" and should
catch cases like 9435 automatically.

Driver for now: Claude interactively through the MCP server, on a handful of
features. Tool signatures must allow a script (headless Claude or a local
model) to drive the same loop later without server changes.

## Protocol

For one feature the agent:

1. Calls `eval_sentences(feature, layer, seed)` and receives about 16
   sentences as text, with ids, in shuffled order. No activations and no
   source tags are returned.
2. Writes predictions from the Neuronpedia label alone, before using any
   other tool on this feature.
3. Investigates freely, then writes an explanation and a second set of
   predictions for the same sentences. It may add up to 4 sentences of its
   own (near-misses, edge cases) to the predictions.
4. Calls `grade_explanation(feature, layer, explanation, predictions)` where
   `predictions` maps a set name (`label`, `investigated`) to a list of
   predictions. The server measures ground truth with the SAE and scores
   every set.

Both sets are graded in one call, so the agent never sees ground truth for
the eval sentences before predicting either set.

Contamination check: for the life of the server process, the backend
records every prompt passed to `run_prompt`, `steer`, `ablate` and
`logit_lens`. At grading time any eval sentence that appears in that record
(exact match after whitespace normalisation) is counted and listed as
`contaminated`. It is reported, not refused. The record is never cleared, so
several features can be evaluated in one session.

## Sentence pool

`eval_sentences(feature, layer, seed=0, n_heldout=4, n_random=8)` builds:

- **heldout**: Neuronpedia examples for the feature beyond the first
  `EXAMPLES_SHOWN = 8` that `feature_info` displays, taken in order, trimmed
  to a window of about 12 tokens before the peak and 4 after (the same window
  `fetch_examples` uses), decoded to plain text without `[[ ]]` markers.
  Fewer than `n_heldout` if Neuronpedia has fewer examples.
- **random**: snippets from the examples of `n_random` other features at the
  same layer, chosen with the seed from features that are not hidden (dense or
  position artifact). One snippet per feature, same window. Real
  pretraining-distribution text, no new dependency, no network beyond the
  per-feature example cache already used by `feature_info`.
- **agent**: sentences present in the agent's predictions but not in the
  pool. Tagged at grading time.

Sentences are shuffled with the seed. The returned structure is
`{"feature", "layer", "seed", "n_heldout", "n_random", "sentences": [{"id",
"text"}]}`. The pool is deterministic in (feature, layer, seed, n_heldout,
n_random), so the server keeps no per-call state beyond the contamination
record; `grade_explanation` takes the same four parameters (same defaults)
and rebuilds the pool to recover source tags. A script driver can therefore
grade after a server restart.

`fetch_examples` currently caps at 8 examples in its cache; it is changed to
cache everything Neuronpedia returns and let callers slice. Existing cache
files written with the old cap are refreshed when they hold fewer than
`EXAMPLES_SHOWN + 1` entries.

## Predictions and scoring

A prediction is `{"text": str, "expected": 0-10, "on": str | null}`.
`expected` is the predicted peak activation relative to the feature's max
(10 = as strong as a top example, 0 = does not fire). `on` is the word the
peak is expected on; optional.

Ground truth per sentence: run it through the model, encode layer `layer`
with the SAE, take the feature's peak activation over positions (excluding
`<bos>`), `rel = act / max_act`, and the peak token.

Scores per prediction set, computed in a new pure module
`src/interp_playground/evals.py`:

- **spearman**: rank correlation between `expected` and actual `rel`. Null
  when either side is constant.
- **firing precision / recall / f1**: actual fires when `rel >= FIRE_REL =
  0.1`; predicted fires when `expected >= PREDICT_FIRE = 2`. Each also broken
  down by source (`heldout`, `random`, `agent`).
- **peak_hit**: over sentences that actually fire and have `on` set, the
  fraction where `on` matches the peak token (case-insensitive, whitespace
  and leading `▁` stripped, prefix match either way so "bridge" matches
  "bridges").
- **rows**: per sentence: id, source, text, expected, on, actual rel, actual
  act, peak token, fired, predicted_fired.

Predictions must cover every pool sentence; missing ones are an error that
names the ids. Texts are matched by id for pool sentences and by text for
agent sentences.

## Outputs

`grade_explanation` returns `{"feature", "layer", "label", "explanation",
"contaminated": [ids], "sets": {name: {scores..., "rows": [...]}}}`.

It also appends one JSON line to `results/explanation_evals.jsonl` (created
on first use, committed to the repo): timestamp, feature, layer, label,
explanation, seed, contaminated, and per set the scores and rows.

`scripts/eval_report.py` reads that file and prints one row per
(feature, timestamp): label f1 and spearman, investigated f1 and spearman,
and the difference. This table is the answer to the question in Purpose.

## Backend changes

- `Backend.eval_sentences(feature, layer, seed, n_heldout, n_random)`.
- `Backend.grade_explanation(feature, layer, explanation, predictions, seed, n_heldout, n_random)`:
  rebuilds the pool, runs all sentences (one forward pass each, under the
  lock), computes ground truth, calls `evals.score_set` per set, appends the
  log line.
- A `_seen_prompts: set[str]` on the backend, appended to by the other
  tools' entry points, never cleared.
- `neuronpedia.fetch_examples` caches all examples; a new
  `example_text(example)` returns the plain-text window.

## Server changes

Two new tools with agent-facing docstrings that spell out the protocol,
including "predict from the label before using any other tool" and "do not
run eval sentences through other tools; contamination is reported".
`eval_sentences` clamps n_heldout to 0-8 and n_random to 0-16;
`grade_explanation` caps prediction sets at 4 and agent sentences at 4 per
set.

## Testing

- `tests/test_evals.py`: spearman on known rankings and constant input;
  precision/recall/f1 by source; peak_hit matching rules; error on missing
  ids.
- Pool construction tested with fake example lists injected through a small
  seam (a function that takes the example lists rather than fetching).
- `scripts/mcp_smoke.py`: call `eval_sentences` for 7272, then
  `grade_explanation` with a trivial prediction set (all zeros) and assert
  the response has scores for it.
- First real run: 7272, 9435 and 9193. Expected from earlier sections: the
  label for 7272 should already score well on firing; 9435's label should
  score poorly on random text containing "San" and the investigated
  explanation ("the token San") should score better; 9193 should show the
  label doing fine, since it was roughly right.

## Out of scope

Per-token predictions, a judge model for thematic steering, a script driver,
cross-layer evals, and any UI. None is precluded by the signatures above.
