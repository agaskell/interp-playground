# Explanation eval: decision log

Dated entries for choices that shape the eval's numbers. Drafted by Claude
during the session; entries marked *proposed* stand until Andy ratifies or
amends them. Anything changed after a run starts gets a new dated entry, never
an edit.

## 2026-11-01 Run 1 (protocol check) — done

- Features 7272, 9435, 9193: previously studied, so the `label` baseline was
  not blind. Treated as a check of the protocol, not a result (FINDINGS 12).
- Defaults: seed 0, n_heldout 4, n_random 8, FIRE_REL 0.1, PREDICT_FIRE 2.

## 2026-11-01 Run 2 (blind) — *proposed*

**Fixes landed first** (tests in `tests/test_evals.py`, `tests/test_backend_pool.py`):
- Random pool keyed on (seed, target feature), so features no longer share
  random sentences.
- Held-out examples spread evenly over the unseen part of the activation
  range (index 8 to the last cached example) instead of the next-best 4.
- Every window puts the peak at a random position (2 to 14 tokens from the
  end), for held-out and random sentences alike, so position carries no hint.

**Feature selection**: `scripts/select_features.py --n 25 --seed 1 --n-heldout 6`
→ `results/blind_features.json`. Criteria: layer 12; never mentioned in
FINDINGS.md or README.md (every integer in those files below 16384 is
excluded, over-inclusively); not dense or a position artifact; has a
Neuronpedia label; at least 14 cached examples. Order is a seeded shuffle of
all 16384 ids; the first 25 that qualify are taken. Nothing was hand-picked
and nothing was skipped by any criterion.

**Pool**: seed 1, n_heldout 6, n_random 10, so 16 pool sentences per feature
plus up to 4 agent sentences in the investigated set. Thresholds unchanged
(FIRE_REL 0.1, PREDICT_FIRE 2).

**Agents** (one chain per feature, same model throughout):
1. Sampler: calls `eval_sentences` once, returns the sentences. Low effort.
2. Label agent: sees only the Neuronpedia label and the sentences, has no
   tool access by instruction, returns the `label` predictions.
3. Investigator: sees the label and the sentences; budget `feature_info`,
   up to 4 `run_prompt` on sentences of its own, exactly one `steer` or
   `ablate`; returns an explanation, the `investigated` predictions and up to
   4 agent sentences, plus the causal result in one sentence.
4. Grader: calls `grade_explanation` once with both sets verbatim, returns
   the scores and the `contaminated` list. Low effort.
A contaminated feature is reported, not dropped.

**Statistic**: per feature, investigated minus label on F1 and Spearman;
report mean delta, wins/ties/losses, and a two-sided exact sign test
(`paired_summary` in `evals.py`, printed by `scripts/eval_report.py`).
Also report `peak_hit` and the by-source numbers. Agent sentences are
excluded from the paired comparison (they exist only in the investigated set)
and reported separately.

**Predictions, before running** (Claude's):
- Investigated beats label on F1 in most features by a small margin: mean
  delta about +0.05, sign test p < 0.05 if ties are few.
- The gain is larger on peak_hit than on F1: labels name topics, not triggers.
- The label does worst on features whose trigger is a token or a format
  (candidates in this set: 9770 "indeed", 3907 end-of-block syntax, 11650
  section starts) when its wording is semantic, and on features whose top
  examples are narrower than the label (7111, 8271).
- At least a third of causal probes come back near null (KL < 0.05).

**Open to Andy**: n features (25), seed (1), pool sizes (6/10), the
tool budget, and whether a null causal probe should count against an
explanation. None of these were tuned after seeing results.

## 2026-11-03 Run 2 (blind) — executed

- Ran as proposed: 25 features, seed 1, n_heldout 6, n_random 10, four
  Claude agents per feature through a Claude Code workflow (100 agents, 13.5
  minutes). Every investigator used the full budget (1 feature_info, 4
  run_prompt, 1 ablate; none chose steer) and wrote 4 agent sentences.
- Two grade calls (6440, 8072) failed because the label agent copied the
  sentences with JSON quotes and the id/text cross-check refused them. Both
  were re-graded by hand from the agents' recorded outputs with only the
  quotes removed (`results/blind_run_agents.json` has the agents' text;
  the regraded records are the last two in `explanation_evals.jsonl`).
  No prediction value was changed.
- Result: F1 +0.12 (15/2/6, p = 0.078), Spearman +0.13 (18/0/5, p = 0.011),
  peak_hit +0.21. Write-up in FINDINGS 13.
- Predictions: direction right, F1 significance missed, peak_hit > F1 gain
  right, 3907/11650 right and 9770 wrong among the worst labels, null-probe
  share badly underestimated (20 of 25, not a third).
- Pool defect found: held-out windows fired in 100 of 150 rows; two features
  had no firing pool row. Proposed fix, not applied: check each candidate
  window fires when building the pool, flag features with fewer than 3
  firing held-out rows. Also: surface grading errors through MCP; format
  sentences as plain text in agent prompts.
