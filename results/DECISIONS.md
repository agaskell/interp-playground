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

## 2026-11-14 Run 4 (extension, three arms) — approved by Andy, set up

Andy approved the proposal above on 2026-11-14, including the third arm.
Written before run 4.

**Selection.** `scripts/select_features.py --n 32 --seed 2 --out
results/blind_features_seed2.json`: same criteria as before, and now also
excluding every feature in an earlier selection file. 32 candidates were drawn
so that refusals could be absorbed; `scripts/check_pools.py --take 25`
(→ `results/blind_pools_seed2.json`) marks the first 25 evaluable in shuffle
order as the batch. All 32 turned out evaluable, so the batch is simply the
first 25 drawn. Nothing was hand-picked.

**Pool.** Version 4 (held-out windows verified to fire; no copy of a shown
example; empty windows skipped, a bug found by this pre-check). Seed 1,
6 held-out + 10 random, thresholds unchanged. Two batch features have short
held-out sets and stay in: 11243 (3 held-out) and 2788 (4).

**Arms**, all Claude Fable 5.2, one chain per feature:
1. `label`: label and sentences only, no tools. Unchanged.
2. `examples` (new): label, sentences and one `feature_info` call. No other
   tool, no agent sentences.
3. `investigated`: unchanged budget (1 `feature_info`, up to 4 `run_prompt`,
   1 causal probe, up to 4 agent sentences).
The examples and investigated agents run side by side and do not see each
other's output. The driver checks the server reports pool version 4 and
stops otherwise.

**Statistics**, all paired on pool sentences, two-sided exact sign test:
- Primary: investigated minus label on firing F1, on this batch alone
  (independent features, clean pool).
- Secondary: examples minus label, and investigated minus examples, on F1
  and Spearman; peak_hit for all three.
- Also reported: investigated minus label pooled over run 3 and run 4
  (48 features), flagged as mixing pool versions 3 and 4.

**Predictions for run 4** (Claude's, before running):
- Primary: label mean F1 about 0.65, investigated about 0.85, gain +0.15 to
  +0.25, sign test p < 0.05.
- The examples arm captures most of the gain: examples minus label about
  +0.13 on F1 and significant on Spearman; investigated minus examples about
  +0.05 on F1 with many ties and not significant, but positive on Spearman.
- The largest label failures will be features whose label is vague
  (13412 "patterns and structures in the text", 12043 "programming-related
  keywords and structures", 3584) and the examples arm will fix most of them.
- At least 19 of 25 causal probes have KL below 0.05.

## 2026-11-14 Run 4 executed

- Ran as set up: 25 features, run tag `blind-4`, pool version 4, three arms,
  Fable 5.2, 125 agents in 11.9 minutes, all graded, nothing contaminated, no
  held-out sentence silent or copied from a shown example. Agents'
  explanations for both tool-using arms: `results/blind4_run_agents.json`.
- **Primary (investigated minus label, firing F1, this batch): +0.09,
  12 wins / 9 ties / 4 losses, p = 0.077. Not significant.** Spearman +0.13
  (p = 0.0002), peak_hit +0.27 (p = 0.0005).
- Secondary: examples minus label F1 +0.08 (12 / 13 / 0, p = 0.0005);
  investigated minus examples F1 +0.01 (5 / 13 / 7, p = 0.77).
- Pooled with run 3 (48 features, pool versions 3 and 4): F1 +0.15,
  25 / 17 / 6, p = 0.0009.
- Budget deviations: three investigators ran three `run_prompt`s; five report
  a call blocked by the permission classifier.
- Exploratory, decided after seeing the data and labelled as such in
  FINDINGS 15: the stratification by the label's own F1, and the sensitivity
  check without the two `<bos>` features (primary +0.07, p = 0.12).
- Predictions: 3 right, 4 wrong. Details in FINDINGS 15.

**Open, for Andy.** Nothing below has been started.
1. Whether to stop collecting here. Recommended: yes for this design. A third
   batch would sharpen the F1 estimate but not change what the two batches
   say, and the design's limits matter more than its n.
2. Whether to fix the three pool defects FINDINGS 15 lists (unreachable
   triggers scored on noise; collapsed whitespace; pool resembles the shown
   examples). Recommended: fix the first two as pool version 5 before any
   further use; treat the third as a design question, not a bug.
3. If the tools question matters for the write-up: a pool the examples do not
   resemble, e.g. sentences sampled from ordinary text at the feature's
   density, or near-misses written by an agent that never sees the arms'
   outputs, predicted by all three arms. This is a new experiment and needs
   its own entry and predictions.

## 2026-11-14 Andy's decisions after run 4

Andy accepted all three recommendations on 2026-11-14:
1. **Stop collecting with the top-example pool design.** Runs 3 and 4 stand as
   the result for that design.
2. **Fix the pool defects** (unreachable triggers scored on noise; collapsed
   whitespace) as pool version 5 before any further use.
3. **Run a new experiment on whether experiments add value beyond reading
   examples**, with a pool the shown examples do not resemble. Design and
   predictions follow in the next entry, written before any run.

## 2026-11-14 Pool version 5

Fixes the two defects Andy approved fixing. Choices made by Claude, for Andy
to confirm or change:

- **Whitespace is kept.** Pool windows keep spaces, newlines and indentation
  as the model saw them, and an agent's own sentences are run exactly as
  written. Text comparisons stay whitespace-insensitive. A predicted peak can
  be a whitespace token.
- **A window must keep half of its recorded activation** (`KEEP_RECORDED =
  0.5`), as well as fire. Neuronpedia records each example's activation in a
  127-token context. A 16-token window that reaches under half of it has lost
  what the feature responds to. Windows that fail are replaced by the nearest
  example that passes, and a feature with fewer than 3 passing windows is
  refused, as before. The 0.5 is a judgement call. It will also refuse
  features that need long context, which narrows what the eval covers.
- **F1 is 0, not undefined, when the feature fires and nothing is predicted.**
  Undefined dropped such features from every paired comparison. Rescoring all
  177 stored prediction sets under the new rule changes none of them, so no
  reported number moves.

## 2026-11-14 Run 5 design: do experiments help on sentences unlike the examples?

Written before the scan finished and before any pool for this run was built.

**Question.** Run 4 found that reading eight top examples gives nearly all of
the gain, on a pool drawn from the same top activations. Does an explanation
built with experiments predict better than one built from examples alone on
sentences that do not resemble those examples?

**New kinds of sentence** (`scripts/scan_wild.py`, `results/wild/`):
- *Wild*: ordinary text where the feature fires. The text is the 16,365
  example contexts in the local Neuronpedia cache, which are top activations
  of about 640 other features, minus the feature's own examples. Firings are
  taken in seeded random order, half from "strong" (0.3 of max or more in
  context) and half from "weak" (0.1 to 0.3). Each window must keep half of
  its in-context activation.
- *Neighbor*: windows around the top examples of the feature's five nearest
  features by decoder cosine, the ones `feature_info` lists. The neighbor
  must fire in the window. What the feature itself does there is not looked
  at when choosing them, so these can be positives or negatives.

**Pool per feature**: 4 held-out, 6 wild, 5 neighbor, 6 random, seed 1,
pool version 5. 21 sentences.

**Features**: the 25 from run 4, minus any that pool version 5 refuses.

**Arms**: the explanations already written in run 4 are reused unchanged
(`results/blind4_run_agents.json`). For each feature, three fresh agents with
no tools predict the new pool: one from the label alone, one from the label
plus the examples-arm explanation, one from the label plus the investigator's
explanation. The two explanation arms get the same prompt and are not told
how the explanation was produced. Fable 5.2. Run tag `blind-5`. Grading is
done by a local script calling the same grading code, not through the MCP
server, since no agent uses a tool.

This differs from runs 2 to 4, where the agent that investigated also
predicted. Here the explanation text has to carry everything. That is a
stricter test of the explanation as an artifact, and it means a difference
from run 4 can come from the pool or from the hand-off.

**Primary comparison**: firing F1 on the 17 new sentences (wild, neighbor,
random), investigated minus examples, paired by feature, two-sided exact sign
test.

**Secondary**: each arm minus label on the same sentences; Spearman and
peak_hit on the same sentences; recall on wild sentences by actual strength
(below 0.3, 0.3 and above); false-positive rate on neighbor sentences where
the feature is silent; F1 on held-out plus random sentences as a check that
the hand-off preserves run 4's ordering.

**Reading the result.** With about 22 features and many ties, this can show
a large effect and cannot rule out a small one. A non-significant primary
means "no large effect seen", nothing stronger.

**Known limits, stated up front**: the explanations were written with run 4's
pool in view and capped at 120 words; one model family writes and reads them;
the wild text is other features' top activations, not a uniform sample;
half the wild sentences are strong by design, which is more than their
natural share; for some features the nearest neighbors are barely similar
(cosine 0.1).

**Claude's predictions.**
1. Pool version 5 refuses 7957 and 11243 and one to three others. 20 to 23
   features are evaluable.
2. The feature fires on about a quarter of neighbor sentences.
3. Held-out plus random sentences: F1 about 0.80 label, 0.87 examples, 0.88
   investigated. The hand-off costs little.
4. New sentences: F1 about 0.55 label, 0.62 examples, 0.68 investigated.
   Primary contrast +0.04 to +0.08, more wins than losses, not significant.
   I put about 30% on it reaching p < 0.05.
5. Spearman on new sentences: investigated minus examples +0.05 to +0.10.
6. Recall on weak wild sentences is below 0.5 for every arm. On strong ones:
   0.6 label, 0.75 examples, 0.8 investigated.
7. False positives on silent neighbor sentences: label most (about 0.3),
   investigated fewest (about 0.15).
