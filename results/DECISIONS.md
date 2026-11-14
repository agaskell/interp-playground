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

## 2026-11-14 Decisions ratified by Andy; rerun (run 3) set up

Andy accepted all six recommendations on 2026-11-14. Written before run 3.

1. **Run-2 setup ratified unchanged** for the rerun: the same 25 selected
   features, seed 1, 6 held-out + 10 random, FIRE_REL 0.1, PREDICT_FIRE 2,
   investigator budget of 1 `feature_info`, 4 `run_prompt`, 1 causal probe,
   4 agent sentences.
2. **Pool fix applied** (pool version 3; tests in `tests/`): every held-out
   window is checked to fire before it is used; a silent one is replaced by
   its nearest firing neighbour in the example list; a feature with fewer
   than 3 firing held-out windows is refused by `eval_sentences`. Window
   length stays 16 tokens: longer windows would leak more context to the
   label agent and change the task.
   Consequence, stated before the run: every held-out sentence in the pool
   now fires, so an agent that knew this could count on about 6 positives in
   16. Neither agent is told the counts or the sources.
3. **Rerun the same features** with the fixed pool. `scripts/check_pools.py`
   (→ `results/blind_pools.json`) shows 23 of 25 evaluable, all with a full
   held-out set. Refused, and so excluded from run 3 before it starts:
   11650 (fires on a mid-text `<bos>`, which plain-text windows cannot carry)
   and 10710 (0 of 34 held-out windows fire). These are the same two features
   that had no firing pool row in run 2, so the paired sample is the same 23.
4. **Primary metric: firing F1**, paired, two-sided exact sign test.
   Spearman and peak_hit are secondary. Chosen now because run 2 had F1 at
   p = 0.078 and Spearman at p = 0.011, and picking after the rerun would be
   cherry-picking.
5. **A null causal probe does not count against an explanation.** Reported,
   not scored.
6. **Extending n**: decided after run 3. If the F1 effect holds, add 25 new
   features with selection seed 2 rather than rerunning these again.

**Changes to the driver, none to the task** (`scripts/blind_eval_workflow.js`,
now in the repo): sentences go into prompts as plain text, one per line with
the id in brackets; agents return ids only and the script attaches each
sentence's text from the sampler's output; grading errors now reach the
grader with their message; every record is tagged `run: "blind-3"` and
`pool_version: 3`.

**Not comparable to run 2 on one axis**: run 2's agents were Claude Fable
5.1; Claude Code now runs Fable 5.2, so run 3's agents are Fable 5.2 in all
four roles. Within run 3 the label and investigated sets use the same model,
so the paired comparison is clean. A difference between run 2 and run 3
confounds the pool fix with the model change and should not be read as the
effect of either alone.

**Predictions for run 3** (Claude's, before running):
- Both sets score higher than in run 2, because the silent "positives" that
  cost recall are gone: label mean F1 about 0.80, investigated about 0.90.
- The paired F1 gain shrinks to about +0.08 but is more consistent: at least
  16 wins of 23 and sign test p < 0.05.
- Investigation still loses on 3 to 5 features, the ones where the label is
  already near perfect and four probes over-broaden the trigger.
- peak_hit gain stays larger than the F1 gain.
- At least 18 of 23 causal probes come back with KL below 0.05.

## 2026-11-14 Run 3 executed; two corrections; decision 6 now due

- Ran as set up: 23 features, run tag `blind-3`, pool version 3, Fable 5.2
  agents, 92 agents in 11.6 minutes, all graded, nothing contaminated.
  Agents' explanations: `results/blind3_run_agents.json`.
- **Primary metric (firing F1, pool sentences, paired): 0.66 to 0.88, +0.22,
  13 wins / 8 ties / 2 losses, two-sided sign test p = 0.007.** Spearman
  +0.19 (20 / 0 / 3, p = 0.0005). peak_hit +0.21.
- **Correction 1: the comparison I ran after run 2 was not the one this log
  specified.** The log said agent sentences are excluded from the paired
  comparison; `paired_summary` used each set's recorded F1, and the
  investigated set's includes its four agent sentences. Fixed in code with a
  test. Run 2 corrected: F1 +0.10, 14 / 5 / 4, p = 0.031 (was reported as
  +0.12, 15 / 2 / 6, p = 0.078). FINDINGS 13 carries a correction note.
- **Correction 2: 9 of 138 held-out sentences in run 3 (11 of 150 in run 2)
  were copies of an example `feature_info` shows.** Excluding them, run 3 is
  F1 +0.23, 13 / 8 / 2, p = 0.007. Pool version 4 excludes such copies when
  building (test added). Run 3's records stay as they are, version 3.
- Budget deviations: 5651 and 8271 used three `run_prompt`s; 3907 had two of
  four denied by the permission classifier. One investigator (7660) used
  `steer`; the rest `ablate`.
- Predictions: 3 right (investigated about 0.90; p < 0.05; at least 18 null
  probes), 5 wrong (label level, size of gain, win count, loss count,
  peak_hit against F1). Details in FINDINGS 14.

**Decision 6 (extend n) is now Andy's to make.** The pre-agreed condition,
"the F1 effect holds", is met. *Proposed*, not started:
- 25 new features, selection seed 2, pool version 4, everything else as
  ratified. About 5 million subagent tokens for two arms.
- *New and optional*: a third arm per feature, an agent that gets the label
  and `feature_info`'s output but no other tool, to separate seeing examples
  from running experiments. Adds roughly a quarter to the cost. If added, the
  primary comparison stays label against investigated; the third arm is
  secondary and its two contrasts are named here before any run: examples
  against label, and investigated against examples.
