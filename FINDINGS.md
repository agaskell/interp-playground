# Findings

Experiment log for the interp-playground tools. Newest at the bottom. Unless
stated otherwise: Gemma 2 2B, Gemma Scope 16k residual SAE at layer 12, fp32
on MPS, greedy decoding.

## 1. Feature 7272 ("bridges"): what it responds to

Script: `scripts/feature_acts.py`.

| Sentence | Peak activation | On token |
|---|---|---|
| We drove across the old **bridge** into town. | 49.9 | bridge |
| The dentist fitted a **bridge** to replace my missing tooth. | 38.7 | bridge |
| In the card game **bridge**, each player bids in turn. | 35.2 | bridge |
| Music can **bridge** cultural divides. | 38.1 | bridge |
| The Golden Gate Bridge opened in 1937. | 40.7 | Bridge |
| Cars queued on the over**pass**. | 21.6 | pass |
| The **ferry** carried us across the river. | 12.3 | ferry |
| The Golden **Gate** is the most photographed landmark... | 8.0 | Gate |

- 35-50: the word "bridge" in any sense. Physical sense scores highest.
- 4-22: bridge-like concepts with no "bridge" token (overpass > ferry > Golden
  Gate > river, crossing). "Golden Gate" scores 7.96 whether or not "Bridge"
  follows, as it must: the model is causal.
- The Neuronpedia label describes the top of the range. Lower activations
  are broader.

## 2. Steering feature 7272, by hand

Script: `scripts/steer.py`. Prompt "My favorite thing to do on weekends is",
residual norm at layer 12 is about 221.

| Scale | Feature 7272 | Random dir, seed 0 | Random dir, seed 1 |
|---|---|---|---|
| 50 | farmers market | beach | beach |
| 70-90 | market, "the smell of the **river**", "**connects** the community" | | |
| 90 | | beach (looping) | mall (looping) |
| 100 | "to go to the old. I love the old... I love the **bridge**" | | |
| 200 | "the to the... bridge" | gym, still English | library (looping) |
| 400 | "bridge bridge bridge" | broken | broken |

- Random directions never produce "bridge" and stay fluent up to about 200;
  the feature direction breaks the text by 100-200. The model is far more
  sensitive to a direction it uses than to an arbitrary one.
- The sweet spot (70-90) steers toward the feature's low-activation concept
  tail (river, connecting), not the word.
- Looping appears with random pushes too, so it isn't evidence of anything.

## 3. Position artifacts (first agent-driven investigation)

Tool: `run_prompt`. Features 1178, 4667, 6889, 11087 fire at 120-380 on
whatever token comes first after `<bos>` ("We", "Bananas", "def", "1", "He")
and nowhere else, including the first word of a second sentence. Density is
about 1/127, i.e. once per Neuronpedia context window. Their labels ("words
that indicate agreement, concession, or logical consequence") are confabulated.

Consequence: `run_prompt` now detects such features per layer with 8 probe
prompts and hides them from rankings; a `features=[...]` watch list returns
exact per-position activations so hypotheses like this can be tested cheaply.

## 4. Steering with metrics (`steer` tool)

Default scales are 1x/2x/4x the feature's max activation (49, 99, 198).
`p_promoted` is next-token mass on the feature's top-10 promoted tokens at the
end of the prompt; `kl` is KL(baseline || run) there.

### 4a. Weekend prompt, again

Identical completions to section 2 (deterministic pipeline). The numbers:

| Scale | KL steered | KL control | "bridge" tokens in completion |
|---|---|---|---|
| 49 | 0.11 | 0.03 | 0 |
| 99 | 1.11 | 0.12 | 2 |
| 198 | 5.55 | 0.63 | 1 |

The feature direction disrupts the distribution 5-10x more than a random one
of the same size. `p_promoted` stays about 0 here because "bridge" is never a
plausible next token right after "is"; `promoted_in_completion` catches the
effect instead. Lesson: measure at the position where the effect can show.

### 4b. A different prompt: "The recipe calls for two cups of"

| Scale | Steered | Control |
|---|---|---|
| 49 | flour, then a maths-problem format ending "The bridge is $" | flour, maths-problem format |
| 99 | "**the gap. The bridge is 100 feet long.**" (x3), KL 2.2 | flour, KL 0.32 |
| 198 | "the gap between the road between the two between..." KL 6.9 | sugar (looping), KL 0.99 |

The effect generalises. At 198 the steered text has no "bridge" token but is
made of the concept's neighbours: gap, between, road, "to the north". Same
pattern as the hand sweep: word at moderate scale, concept fragments beyond.

### 4c. Negative steering: "We drove across the old"

Baseline: next-token mass on bridge tokens is **0.32**; completion "bridge to
the new bridge. The old bridge is still there".

| Scale | p(bridge tokens) | KL | Steered completion | Control p(bridge) |
|---|---|---|---|---|
| -25 | 0.090 | 0.59 | "bridge and then turned right on the road to the beach." | 0.29 |
| -50 | 0.0007 | 2.90 | "**country**,\nWe drove across the old country,\nWe" | 0.25 |
| -100 | 0 | 8.96 | "-ness of the 1970s and" | 0.18 |
| -200 | 0 | 12.4 | newlines | 0.013 |

- Subtracting half the feature's natural max removes "bridge" almost entirely
  (0.32 to 0.0007) while the control barely moves (0.32 to 0.25). The feature
  is load-bearing for this prediction, not just correlated with it.
- What replaces it: "country". Not a synonym and not road/river; something
  else one "drives across the old ___". The model falls back to the next
  plausible continuation rather than a near-synonym. (Prediction before the
  run: "something nearby, maybe not road or river". Held.)
- Beyond -50 the model breaks faster than with positive steering of the same
  size (-100: KL 9.0 vs +99: KL 1.1 in 4a). Pushing against a direction the
  model is actively using at this position is more disruptive than adding
  one it isn't.

## 5. Ablation vs. negative steering: where a direction matters

Tools: `ablate`, `steer` (with `positions`), `logit_lens`. Prompt "We drove
across the old"; baseline p(" bridge") = 0.31.

**Zero-ablating 7272 does nothing here** (KL 0.0005): at "old" the feature is
inactive, so there is nothing to remove. Ablating each of the 12 most active
layer-12 features at "old" also barely moves p(bridge); the largest effect
was in the other direction (removing 1127 "old" *raised* p(bridge) to 0.39,
presumably by removing competition from "old-fashioned"-style continuations).
No single layer-12 feature at that position carries the prediction.

**Yet subtracting the 7272 direction at scale -50 removes "bridge" almost
entirely** (section 4c). Splitting by position:

| Subtract 50 x direction at... | p(bridge) | top next token |
|---|---|---|
| nothing (baseline) | 0.31 | bridge |
| `<bos>` only | 0.28 | bridge |
| last position ("old") only | 0.16 | bridge |
| earlier positions ("We drove across the") only | 0.03 | town / country |
| all positions | 0.0007 | country |

The direction matters most at the *context* positions, where the SAE encoder
says the feature is off (0, except 4.06 at "the"). The logit lens agrees on
timing: p(bridge) at "old" is about 0 through layer 18, 0.09 at layer 19 and
0.62 at layer 20, so the prediction is assembled after layer 12 by heads
reading the earlier positions. What they read there is aligned with 7272's
decoder direction even though its encoder threshold wasn't crossed.

Takeaways:
- "Feature X is inactive" (encoder) and "direction X is unused" (decoder) are
  different claims. Ablation tests the first; steering tests the second.
- An agent that only ablates would conclude 7272 is irrelevant to predicting
  "bridge". An agent that only steers would conclude it's essential. Both
  tools are needed, and `positions` makes the distinction visible.
- The last two layers *reduce* confidence: p(bridge) peaks at 0.89 (layer 23)
  then falls to 0.14 (layer 24) and 0.31 (layer 25) as "wooden", "stone" and
  "old" come back. Late layers calibrate rather than commit.

## 6. Logit lens trajectories

| Prompt | Final prediction | First layer with p >= 0.5 | Notes |
|---|---|---|---|
| The Eiffel Tower is located in the city of | Paris (0.95) | 20 | 0.17 at layer 19, 0.93 at 20: one layer does most of the work. Layer 18's top token is " city". |
| We drove across the old | bridge (0.31) | 20 | See section 5. |
| The capital of Japan is | a (0.18) | never | The base model hedges ("is a ..."); " Tokyo" is not the top guess at any layer. |
| She opened the door and walked | in (0.35) | never | Open-ended continuation; no layer is confident. |

In layers 6-18 the lens mostly shows the *current input token* (" old" 0.82
at layer 12, " is" 0.95, " walked" 0.48): the residual stream there still
encodes "what this token is" more than "what comes next". Mid-layer lens
output is not a prediction; read `p_final` instead.

## 7. Second label check: 9435 "San Francisco and its surrounding areas"

Tools: `feature_info`, `run_prompt` with `features=[9435]`.

- Promotes: " San", "San", " SAN", " san". Nearest feature by decoder
  direction: 11435 "saints" (cos 0.30). Top examples include "[[San]]itize a
  URL" and "[[San]]ofi-Aventis".
- Activations: 32-35 on the token "San" in "San Francisco", "San Diego",
  "San Antonio", "**San**itize" and "Sanofi"; 6-13 on the token after it;
  **0** on "The Golden Gate Bridge and Alcatraz are in the Bay Area" (3.7 on
  " Bay") and on "Oakland is across the bay from the city".

It is a "San" string feature, not a San Francisco feature. The label was
written from top examples that happened to mention the city.

Running tally of label checks: 7272 roughly right but too narrow (section 1);
four position artifacts with invented meanings (section 3); 9435 wrong. The
auto-generated labels are useful search keys and poor ground truth.

## 8. Concept features steer gracefully; token features don't

Tool: `steer`, default scales (1x/2x/4x max_act), weekend prompt, random
control at each scale. Chosen from label search plus `feature_info`:

- 5608 "dogs": promotes spellings of "dog". A token feature like 7272.
- 9193 "grief and loss": promotes " death", " funeral", " Mortuary";
  nearest features are "burial sites" and "casualties". A concept feature.

| Scale | 5608 dogs (token) | 9193 grief (concept) |
|---|---|---|
| 1x | beach (unchanged) | beach (unchanged) |
| 2x | "to take my dog" then zero-width-space garbage, KL 0.5 | "to go to the **cemetery and visit my loved ones**. I'm not a fan of the holidays, but I'm grateful..." KL 0.24 |
| 4x | garbage, KL 2.4 | "to **honor my loved ones. I'm not a stranger to grief**, and I'm not a stranger to loss." KL 1.3 |

Controls stayed on beach/gym (KL 0.02-0.5). On "The weather this morning
was", grief at scale 40 (KL 0.05, barely above control's 0.03) gave "I took a
walk with my friend, and we walked to the cemetery."

- Token-level features (bridges, dogs) have a cliff: no effect, then the word,
  then collapse. Concept features bend the whole text while it stays fluent.
  This matches the Golden Gate Claude result and explains why 7272 was a poor
  steering target.
- `p_promoted` was 0.000 in every grief run: the model never says "death"
  right after "is", it writes about cemeteries and loved ones instead.
  Lexical metrics miss thematic steering. `steer` now takes `target_tokens`
  so an agent can measure the theme it expects; a judge model would be the
  next step.

## 9. Logit lens: facts are retrieved before the output token is chosen

Tool: `logit_lens` via `scripts/tool.py` (Andy ran these). Compare two prompts
that differ by one cue word.

**"Romeo and Juliet was written by William"** → " Shakespeare" (0.98).
p_final: 0.005 at layer 17, 0.37 at 18, 0.56 at 19, then 0.71, 0.73, 0.79,
0.89, 0.98 through layer 24. No descriptor stage; layer 17's top token is
still " William". At layers 18-22 the mass is split between " Shakespeare"
and "Shakespeare" (no space), e.g. 0.71 vs 0.27 at layer 20: the model knows
*who* several layers before it settles *which token*.

**"Romeo and Juliet was written by"** → " William" (0.66), " Shakespeare" (0.18).

| Layers | Leading token |
|---|---|
| 15-18 | " famous" (0.27 → 0.58), " renowned" |
| 19 | " writers", " writer" |
| 20 | " **Shakespeare**" 0.35 |
| 21-23 | " the", " a", " play" |
| 24-25 | " **William**" 0.22 → 0.66 |

- The fact (" Shakespeare") is retrieved at layer 20, the same layer as
  " Paris" in section 6. The last five layers turn it into the right first
  token of the name. `p_final` tracks " William" and is ~0 until layer 24, so
  **p_final measures when the model commits to a token, not when it knows the
  answer.** Read the top tokens from layer 15 on; the docstring now says so.
- The cue word made recall two layers earlier (18 vs 20) and firmer (0.98 vs
  0.66). Some of the first run was name completion, not fact retrieval.
- The descriptor stage (famous → writers → Shakespeare, like city → Paris)
  appears when nothing has narrowed the answer's form and is skipped when a
  cue has. Hypothesis from two prompt pairs; cheap to test on more.
- No late-layer confidence dip for a confident fact (0.98 holds at 24-25).
  The dip in section 5 (0.89 → 0.31 for "bridge") happened where the model had
  real alternatives.

## 10. Measuring thematic steering across the completion

Tool: `steer`, feature 9193 (grief), weekend prompt, `target_tokens`
cemetery/funeral/grief/loss. Prediction beforehand: `p_target` at the
prompt's end stays ~0 (nobody writes "weekends is cemetery"), the completion
count rises. Held:

| Scale | p_target at prompt end | target tokens in completion | mean p_target over completion | control mean |
|---|---|---|---|---|
| 36 (1x) | 0.000 | 0 | 0.003 | 0.000 |
| 73 (2x) | 0.0001 | 1 (cemetery) | 0.016 | 0.000 |
| 145 (4x) | 0.0005 | 2 (grief, loss) | 0.030 | 0.000 |

The count undercounts: the 2x completion ("the cemetery and visit my loved
ones") and 4x ("honor my loved ones... I've lost loved") are about grief
throughout. So `steer` now re-runs prompt + completion in one teacher-forced
pass (with a mask reproducing the generation-time `positions` edit; argmax
matched the generated token at every step in all three modes) and reports
the **mean next-token mass on promoted/target tokens over the completion**
(`p_promoted_completion`, `p_target_completion`). It is graded where the
count is 0/1/2, and it separates steered from control already at 1x (0.003
vs 0.000), where KL cannot (0.019 vs 0.014). Still lexical: a judge model is
the next step for "loved ones".

Side result, scale 73, 20 tokens: steering only the **generated** positions
(`positions="last"`) still gives "the cemetery and visit my family" (mean
p_target 0.017, KL at prompt end only 0.03); steering only the **prompt**
gives beach (0.008, no target tokens). The opposite of section 5, where the
context positions carried the effect. Where a direction matters depends on
whether the output is being *predicted from context* (bridge) or
*generated along a theme* (grief).

Also: 1x max_act added at every position barely moves KL above the control.
Steering needs about twice a feature's natural peak.

## 11. Ablating 1127 "old" on "We drove across the old"

Tool: `ablate`, top_k 8. 1127 fires at 46.2 (max 51.8) on "old" and promotes
" old", " Old", " fashioned". Zeroing it: KL 0.43 (vs 0.0005 for 7272 at the
same spot; ablation is informative exactly where the feature fires).

| Next token | Baseline | Ablated |
|---|---|---|
| bridge | 0.308 | 0.391 |
| suspension | 0.030 | 0.118 |
| wooden | 0.047 | 0.021 |
| stone / iron / town | 0.034 / 0.022 / 0.031 | out of top 8 |
| old | out of top 8 | 0.013 |

- Prediction ("fashioned" loses mass) was wrong: it was never in the top 8.
  The competition story survived with different competitors. The feature
  carries the *meaning* of "old"; remove it and the continuations that depend
  on it (old **wooden**/**stone**/**iron** bridge) fall, a modern bridge type
  (suspension) quadruples, and "bridge" rises because there is less reason to
  add a second adjective.
- " old" reappears as a next token: without the feature the model no longer
  fully registers that it just said "old". Direct evidence the feature encodes
  "this token is old", not a vaguer sense of age.
- Check the token list, not just the headline number; the mechanism was in
  the losers, not the winner.

## 12. First explanation eval: 7272, 9435, 9193

The `eval_sentences` / `grade_explanation` protocol (see README) run once
per feature by Claude through the MCP server, 2026-11-01. Pool per feature:
4 held-out Neuronpedia examples the agent had not seen plus 8 random windows
from other features, seed 0. Two prediction sets each: `label` written from
the Neuronpedia label before any tool call, `investigated` written after
`feature_info`, three or four `run_prompt`s on fresh sentences and one
`ablate`, plus four agent-written sentences. Raw records are in
`results/explanation_evals.jsonl`; `scripts/eval_report.py` prints the table.

Predictions, written before running: 7272's label would already score well
and investigation would only help on near-misses; 9435's label would fail
on anything "San" that is not San Francisco; 9193's label would do fine and
investigation would add little.

| feature | set | n | F1 | recall | Spearman | peak_hit |
|---|---|---|---|---|---|---|
| 7272 bridges | label | 12 | 1.00 | 1.00 | 0.96 | 1.00 |
| | investigated | 16 | 1.00 | 1.00 | 0.97 | 0.88 |
| 9435 "San Francisco" | label | 12 | 0.89 | 0.80 | 0.94 | 0.00 |
| | investigated | 16 | 0.94 | 0.89 | 0.97 | 1.00 |
| 9193 grief | label | 11 | 0.86 | 0.75 | 0.81 | 1.00 |
| | investigated | 15 | 0.92 | 0.86 | 0.88 | 0.83 |

Precision was 1.0 everywhere and nothing was contaminated.

**7272.** Both sets saturate on the pool. Every held-out window contains the
literal token "bridge" and no random window contains anything bridge-like,
so for a token-level feature this pool cannot separate a label from an
explanation. The signal was in the agent rows: viaduct fired at 0.41 and a
guitar bridge at 0.74, both as predicted; the ferry sentence fired at 0.25
on " ferry" rather than on "crosses", so the weak tail covers crossing
vehicles as well as spanning verbs. Causal check positive: ablating the
feature after "the famous bridge" removes " over" from the next-token
distribution (KL 0.19).

**9435.** The label failed the way section 7 predicted. It put the peak on
the wrong token in all four held-out rows (the peak is on " San", never on
the city), and it missed a random-pool sentence containing "Sanchez" that
fired at 0.33. The investigated explanation ("orthographic: the token San
and words starting san/sant/são") called Sanchez, sanitize(), Sanofi and
Sandra correctly. One surprise: a Bay Area sentence with no "San" in it
fired at 0.17 on " Palo", so there is a small genuine place component under
the spelling detector. Causal check near null: ablating at "Boston for San"
leaves the Francisco/Diego/Jose split almost unchanged (KL 0.04). The
feature detects the spelling and does not carry the city's identity.

**9193.** Label and investigation agree on what fires; investigation added
calibration. Bereavement-specific words peak (sorry for your "loss" 1.0,
grieving 0.84, sorrow 0.80, grief 0.76, widowed 0.75) and general death
words fire at half that (mourning 0.53, passed away 0.49, funeral 0.38,
cemetery 0.30). "Loss" is disambiguated: a net loss of $4 million fired 0,
a lost football match fired 0.25 on " lost". A wedding thank-you with
"love and support" was near zero. The label's one miss was a random window
(".Vale de Lua") that fired at 0.16 on "Vale", which neither set predicted.
Causal check null: ablating across "Since his wife passed away last spring,
he has been" changed the continuation by KL 0.005.

**About the eval itself**, which is what this first run was for:

- The held-out windows are centred on the peak token, so a sentence that
  contains the trigger word is easy to spot and `peak_hit` is easier than it
  looks. For token-level features the pool saturates; the agent sentences and
  the random windows that happen to contain near-misses (Sanchez, Vale) carry
  the discriminating signal. A harder pool would draw held-out examples from
  lower in the activation range, not only the top.
- The random windows are identical across the three features because the
  seed alone picks the random features. That is a bug to fix (mix the target
  feature into the seed): the agent learns the random set after one feature.
- The `label` baseline is only clean for features the agent has never
  investigated. For 9435 the label predictions were written by a mechanical
  rule because section 7 already existed; a fair baseline needs features
  chosen blind.
- Three causal checks, one positive (7272) and two near null (9435, 9193).
  A layer-12 feature can be a reliable detector and still have no measurable
  effect on the next token at the position tested. The ablation result is
  reported in the explanation but not scored, which is the right place for it
  until there is a defensible way to score "causal role".
- n=3 features, all previously studied. The deltas (+0.00, +0.05, +0.07 F1)
  are the direction predicted, not a result. A real test is 20-30 features
  the agent has not seen, with the same protocol driven by a script.

## 13. Blind explanation eval: 25 features, label vs investigation

Run 2026-11-03 on 25 layer-12 features chosen by a seeded shuffle and never
mentioned in this repo before (`results/blind_features.json`; criteria and
pre-registered predictions in `results/DECISIONS.md`). Pool per feature: 6
held-out Neuronpedia examples spread over the unseen part of the activation
range plus 10 random windows from other features, peak at a random position
in every window, 16 sentences in all. One chain of four Claude agents per
feature: a sampler, a tool-less label agent that saw only the Neuronpedia
label and the sentences, an investigator with a fixed budget (`feature_info`
once, four `run_prompt`s on its own sentences, one causal probe, four agent
sentences), and a grader. 100 agents, 13.5 minutes, nothing contaminated.
Two grades failed on a transcription slip (the label agent kept JSON quotes
around the sentences, and the id/text cross-check refused the call) and were
re-run by hand with the quotes removed and nothing else changed.

> **Correction, 2026-11-14.** The table below scored the investigated set
> over its pool sentences *plus* the agent's own four sentences, which the
> label set does not have. On the 16 shared pool sentences, as the decision
> log specified, run 2 was F1 0.72 to 0.82 (+0.10, 14 wins / 5 ties / 4
> losses, sign test p = 0.031) and Spearman 0.68 to 0.80 (+0.13, 20 / 0 / 3,
> p = 0.0005). `paired_summary` now compares pool sentences only. See
> section 14.

**Paired result, investigated minus label** (23 features with both scores):

| metric | label mean | investigated mean | mean delta | wins / ties / losses | sign test |
|---|---|---|---|---|---|
| firing F1 | 0.72 | 0.85 | +0.12 | 15 / 2 / 6 | p = 0.078 |
| Spearman | 0.68 | 0.82 | +0.13 | 18 / 0 / 5 | p = 0.011 |
| peak_hit | 0.43 | 0.64 | +0.21 | 12 / 6 / 4 | |

Pooled over all 400 pool rows: label precision 0.68 and recall 0.73,
investigated 0.78 and 0.86. False alarms on random windows fell from 29 to
19 and misses on held-out windows from 23 to 13. Agent-written sentences (100
in all): 60 fired, the investigator predicted firing on 53, 50 were right,
and it named the peak token on 32 of those 50.

**Where the label failed** (label F1 at or below 0.6 for 9 of 25):

- 8072, labelled "scientific processes or methodologies involving
  measurements and statistics", is a syntactic feature: it fires on the first
  token of a modifier attached to a noun phrase inside an embedded clause
  ("the documents [found] in", "accused [of] theft"). Label F1 0.00; the
  investigated explanation got 0.47, still poor because the trigger is
  subtle and the feature is weak (max activation about 15).
- 11650, "the start of new sections or paragraphs", fires only on a `<bos>`
  token in the middle of a packed sequence, never on prose. The label agent
  predicted firing on headers and got every one wrong; the investigator
  planted a mid-text `<bos>` in an agent sentence and it fired at 0.85.
- 6440, "code elements and syntax related to GUI frameworks", peaks on the
  dot after `System.Windows.Forms` and barely elsewhere.
- 10354 (0.44 to 1.00), 7660 (0.40 to 0.82), 13026 (0.57 to 0.92), 9605
  (0.57 to 0.86) and 3907 (0.55 to 0.71): labels that name a topic the
  feature is only partly about. 9605 "historical naval terminology" is really
  aircraft and ship type designations (Boeing, F-35, Leander-class).

**Where investigation lost** (6 features): 16293, 5651 and 8316 went from a
perfect label F1 to 0.83-0.86, and 6184, 4344 and 8271 slipped by 0.03-0.09.
In each case the investigator broadened the trigger from four probes and
predicted firing on random windows that stayed silent. Four probes are
enough to find a trigger and not enough to bound it.

**Causal probes.** Every investigator chose `ablate`. 20 of 25 came back
with KL below 0.05, three between 0.05 and 0.2, and two above: 5323
(contested territory; ablation cut p("territory") from 0.63 to 0.40, KL 0.25)
and 339 (KL 0.20). A layer-12 feature that is a clean detector is usually
not something the next-token distribution depends on at the position tested.

**Predictions scored.** Mean F1 gain was predicted at +0.05 and came in at
+0.12, but the F1 sign test did not reach 0.05 (Spearman did). The peak_hit
gain was larger than the F1 gain, as predicted. 3907 and 11650 were among the
worst labels as predicted; 9770 "indeed" was not, because its label names the
token, and 7111 gained as predicted while 8271 did not. "At least a third of
probes near null" undershot badly: four in five were.

**What the run says about the eval itself:**

- Held-out windows fired in only 100 of 150 cases. Neuronpedia's activations
  come from 128-token contexts and a 16-token window loses the context a
  contextual feature needs, so a third of the "positives" are negatives in
  the pool, and an agent that treats them as positives is penalised. The pool
  builder should check each candidate window fires and skip the ones that
  do not; two features (10710, 11650) had no firing pool row at all and
  should have been flagged at `eval_sentences` time.
- Random windows fired in 8 of 250, so random precision is a weak signal and
  the agent sentences carry most of the negative-case information.
- The MCP layer swallowed the grading error ("Error executing tool
  grade_explanation"), so the grader could not see that the id/text check had
  fired. The server should return the message.
- Formatting pool sentences as JSON strings in a prompt made two of 25 label
  agents copy the quotes. Plain text with a delimiter is safer.
- The label baseline is a fair opponent now (blind features, no tools), and
  it is not weak: mean F1 0.72 from a one-line auto-generated label. The gain
  from investigation is real on ranking and peak location, modest and not yet
  significant on firing F1 at n = 23.

## 14. Blind eval rerun with a fixed pool: the gain doubles

Run 2026-11-14, tagged `blind-3` in `results/explanation_evals.jsonl`. Same
selection, seed, pool sizes, thresholds and investigator budget as section 13,
all ratified beforehand, with firing F1 named as the primary metric and
predictions written down first (`results/DECISIONS.md`). Two things differ
from section 13:

- **Pool version 3.** Every held-out window is checked to fire before it is
  used; a silent one is replaced by its nearest firing neighbour. Two of the
  25 features (11650, 10710) have no held-out window that fires in 16 tokens
  and were excluded before the run, leaving 23.
- **The agents are Claude Fable 5.2** in all four roles; section 13's were
  Fable 5.1. The label-versus-investigated comparison inside this run is
  clean. The difference between this run and section 13 mixes the pool fix
  with the model change.

92 agents, 11.6 minutes, all 23 graded, nothing contaminated.

**Primary result, investigated minus label, on the 16 pool sentences both
sets predicted** (23 features):

| metric | label | investigated | mean delta | wins / ties / losses | sign test |
|---|---|---|---|---|---|
| firing F1 (primary) | 0.66 | 0.88 | +0.22 | 13 / 8 / 2 | p = 0.007 |
| Spearman | 0.69 | 0.88 | +0.19 | 20 / 0 / 3 | p = 0.0005 |
| peak_hit | 0.42 | 0.63 | +0.21 | 10 / 9 / 2 | |

Pooled over the 368 pool rows: label precision 0.84 and recall 0.61,
investigated 0.94 and 0.85. False alarms on random windows 16 against 8;
missed held-out windows 51 against 19.

**Two defects found while analysing, both corrected and neither changes the
conclusion:**

- The recorded set-level F1 for the investigated set includes the agent's own
  sentences. With them in, the numbers look better than the table above
  (F1 +0.24, 20 / 2 / 1, p = 0.00002). The table above is the pre-registered
  comparison; `paired_summary` and `scripts/eval_report.py` now use it.
- Nine of the 138 held-out sentences were copies of one of the eight examples
  `feature_info` shows, because Neuronpedia's example lists repeat texts. The
  investigator had seen those with the peak marked. Dropping them: F1 0.64 to
  0.87, +0.23, 13 / 8 / 2, p = 0.007. Pool version 4 never holds out a copy
  of a shown example.

**Where the gain comes from.** Recall on held-out sentences, by how strongly
the feature actually fired:

| actual activation (fraction of max) | n | label recall | investigated recall |
|---|---|---|---|
| 0.1 to 0.3 | 47 | 0.51 | 0.74 |
| 0.3 to 0.6 | 35 | 0.49 | 0.86 |
| above 0.6 | 56 | 0.82 | 0.96 |

A label describes the top of a feature's range. Below 0.6 of max the label
agent finds half the firings; the investigator, who has measured weaker
cases, finds three quarters or more.

**Features**, pool F1 label to investigated:

- **7660** "references to a crisis or challenging societal situations",
  0.00 to 0.91. It is a Spanish-language detector: fires on nearly every
  token of Spanish text, on Hispanic names in English at half strength, weakly
  on Portuguese, never on English crisis text. The one investigator that
  chose `steer` used it here: at 2.6 times max the continuation of an English
  prompt switched into Spanish (KL 0.31 against 0.05 for a random direction).
- **8072** "scientific processes ... measurements and statistics", 0.00 to
  0.77. The syntactic feature from section 13, found again by a different
  model: fires inside a long noun phrase while the clause still owes
  something after it.
- **9605** "historical naval terminology", 0.29 to 1.00: military and
  aviation hardware types (Boeing, McDonnell Douglas, Leander-class).
- **14431** "legal and procedural language related to eligibility", 0.22 to
  0.73: the hinge between a satisfied condition and its consequence ("if the
  documents are complete[,] ..."), not legal vocabulary.
- **13026** 0.44 to 1.00, **10354** 0.40 to 0.92, **6440** 0.60 to 1.00.
- Eight ties, where both agents made the same firing calls, and two losses:
  **4344** (0.83 to 0.67), where the investigator narrowed the trigger to the
  `</summary>` tag and missed other doc-comment text, and 7111 by 0.01.

**Agent-written sentences** (92): 56 fired; the investigator predicted
firing on 50 and was right on all 50; it named the peak token on 34.

**Causal probes.** 22 ablations and one steer. 18 of 23 had KL below 0.05,
three between 0.05 and 0.2, two above (339 at 0.47; 7660's steer at 0.31).
Same picture as section 13: most clean layer-12 detectors leave the next
token alone at the position tested.

**Budget deviations.** Three investigators ran fewer than four `run_prompt`s:
5651 and 8271 used three, and 3907 had two of its four denied by Claude
Code's permission classifier. All used `feature_info` once and one probe.

**Predictions scored.** Right: investigated mean F1 about 0.90 (0.88); the
sign test under 0.05; at least 18 null probes (exactly 18). Wrong: the label
would rise to about 0.80 (it fell to 0.66, because the fixed pool swaps silent
windows for weakly firing ones, which is where labels fail); the gain would
shrink to +0.08 (it was +0.22); at least 16 wins (13, with 8 ties);
investigation would lose on 3 to 5 features (2); the peak_hit gain would
exceed the F1 gain (about equal).

**What this does and does not show.**

- On these 23 features, an agent that looks at a feature's examples and runs
  five experiments predicts its firing better than the same agent reading
  the auto-generated label, and the difference is largest on weak firings and
  on labels that name a topic when the trigger is a language, a syntactic
  slot or a formatting pattern.
- It does not separate *looking* from *experimenting*. `feature_info` shows
  eight top examples from the same source as the held-out sentences. A third
  arm that gets the label and those examples but no probes would say how much
  of the gain needs tools at all.
- One model family writes the explanations and makes the predictions, one SAE
  at one layer, 23 features, one seed. The explanation's quality is measured
  only through the predictions its author makes, not by handing the
  explanation to a separate simulator.

## 15. Three arms on 25 new features: the examples do most of the work

Run 2026-11-14, tagged `blind-4`. 25 new layer-12 features drawn with
selection seed 2 (`results/blind_features_seed2.json`, the first 25 of
`results/blind_pools_seed2.json`), pool version 4, Claude Fable 5.2 in every
role. Setup, statistics and predictions were written down first
(`results/DECISIONS.md`). New in this run: a third arm, `examples`, that gets
the label, the sentences and one `feature_info` call, and runs no experiments.
125 agents, 11.9 minutes, all 25 graded, nothing contaminated, every held-out
sentence fired and none was a copy of a shown example.

**Results on the pool sentences all three arms predicted** (25 features):

| arm | firing F1 | precision / recall (pooled rows) | Spearman | peak_hit |
|---|---|---|---|---|
| label | 0.81 | 0.84 / 0.81 | 0.79 | 0.48 |
| examples | 0.89 | 0.96 / 0.84 | 0.87 | 0.61 |
| investigated | 0.90 | 0.96 / 0.86 | 0.92 | 0.75 |

| paired contrast | F1 delta | wins / ties / losses | sign test | Spearman delta, test | peak_hit delta, test |
|---|---|---|---|---|---|
| **investigated minus label (primary)** | +0.08 | 12 / 9 / 4 | **p = 0.077** | +0.13, p = 0.0002 | +0.27, p = 0.0005 |
| examples minus label | +0.07 | 12 / 13 / 0 | p = 0.0005 | +0.09, p = 0.002 | +0.13, p = 0.15 |
| investigated minus examples | +0.01 | 5 / 13 / 7 | p = 0.77 | +0.05, p = 0.83 | +0.14, p = 0.42 |
| investigated minus label, sections 14 and 15 pooled (48) | +0.15 | 25 / 17 / 6 | p = 0.0009 | +0.16, p < 0.0001 | +0.25, p < 0.0001 |

*Rounding corrected 2026-11-15:* the first two F1 deltas were printed as
+0.09 and +0.08. They are 0.085 and 0.075.

**The primary comparison did not reach significance in this batch.** The
direction is the same as section 14, the size is under half (+0.08 against
+0.22), and four features got worse. Pooled over both batches it holds.

**Reading the examples accounts for nearly all of the firing gain.** The
examples arm never did worse than the label (12 wins, 13 ties, no losses) and
experiments added +0.01 F1 on top, with more losses than wins. The one place
the investigator looks better than the examples arm is naming the peak token
(0.75 against 0.61), and that difference is not significant either.

**Why this batch differs from section 14** (exploratory, not pre-registered).
The gain depends on how bad the label is, and this batch had better labels:

| label's own pool F1 | features (both batches) | mean F1 gain from investigating | wins / ties / losses |
|---|---|---|---|
| below 0.6 | 10 | +0.53 | 8 / 2 / 0 |
| 0.6 to 0.9 | 18 | +0.10 | 11 / 5 / 2 |
| 0.9 and above | 20 | 0.00 | 6 / 10 / 4 |

Section 14's batch had 7 of 23 features with a label F1 below 0.6; this one
had 3 of 25. Where the label already works, investigating does not help and
sometimes over-narrows. Section 14's other pattern, a large recall gain on
weak firings, did not repeat: recall on held-out sentences firing below 0.3 of
max was 0.68, 0.70 and 0.68 for the three arms.

**Features.**

- **312** "counterfactual reasoning and concepts in quantum mechanics": F1
  0.25, 0.50, 1.00. Fires across expository prose about abstract mathematical
  objects (sample space, distribution, eigenspace). All eight shown examples
  were the same text, so the examples arm could only guess "space"; this is
  the case where experiments earned their keep.
- **2616** is the token " over" in every sense; **16333** is the word
  "majority"; **4409** is the copyright symbol after the word "Copyright".
  These three had the only large causal effects: ablating 4409 after
  "Copyright ©" dropped the next token's probability from 0.999 to 0.17
  (KL 1.79); ablating 2616 on "the same song over and" turned " over" into
  " again" (KL 0.98); ablating 16333 halved " minority" after "the majority
  held ... while the" (KL 0.22).
- **7957** "viruses and diseases" and **11243** "punctuation marks" are both
  mid-text `<bos>` detectors. With 11650 from the first batch that makes
  three in 50 sampled features, each under a topical label.
  Both investigators said so and scored badly anyway (7957: F1 0.29), which
  is a defect of the eval, below.
- **276** (0.91 to 0.67), **2011**, **4190**, **3584**: investigation lost by
  narrowing a trigger the label had right.

**Causal probes.** All 25 investigators chose `ablate`. 20 had KL below 0.05.

**Budget deviations.** Three investigators (16333, 11243, 2616) ran three
`run_prompt`s. Five report an attempt blocked by Claude Code's permission
classifier as too close to a pool sentence; for three of those it was an
extra call beyond the four allowed. Every examples agent made exactly one
`feature_info` call.

**Predictions scored.** Right: the examples arm captures most of the gain;
investigated minus examples is small, tie-heavy and not significant on F1; at
least 19 null probes (20). Wrong: label mean F1 about 0.65 (0.81); primary
gain +0.15 to +0.25 (+0.08); primary significant (p = 0.077); the vague
labels would be the worst (13412 scored 1.00; the worst were 312, 7957 and
11243).

**Defects in the eval this run exposed, none fixed yet:**

- *A feature whose real trigger cannot appear in the pool is still accepted
  if its weak tail fires.* The windows are plain text and cannot carry a
  mid-text `<bos>`. 7957's held-out sentences fired at 0.12 to 0.38 of max on
  tokens like " Mann" and "IBC": noise. An agent that correctly says "this is
  a `<bos>` detector" predicts zero and is marked wrong. Without those two
  features the primary contrast is +0.07 (11 / 8 / 4, p = 0.12) and examples
  minus label +0.06 (10 / 13 / 0, p = 0.002). A fix would compare each
  window's activation with what Neuronpedia recorded for that example and
  refuse features whose windows keep only a fraction of it.
  *Correction, 2026-11-15:* "noise" was wrong. Neuronpedia's example list
  spans a feature's whole activation range, and 7957's examples from about
  rank 12 down are weak firings on ordinary tokens, recorded at 18 and below
  against a maximum of 65. The windows reproduced those activations. The two
  features were scored on their real weak tail. The proposed fix was built
  (pool version 5) and, correctly, does not refuse them. See section 16.
- *Whitespace is collapsed*, in pool windows and in the agent's own
  sentences. 10140 fires on the indentation after a Python class header; its
  held-out windows, with the newlines gone, fired at 0.13 to 0.25, and the
  investigator's test sentence lost its newlines before it was run. A peak
  named as a whitespace token is also dropped.
- *The pool resembles what the examples arm is shown.* Held-out sentences
  come from the same list of top activations as the eight examples, so this
  eval measures "can you recognise more of the same" and is close to the best
  case for the examples arm. Whether experiments help on sentences unlike the
  top examples (near-misses, other senses, weak firings in ordinary text) is
  not measured: only the investigator predicts its own near-miss sentences.

**Where this leaves the question.** On 48 blind features, an agent that
studies a feature beats the same agent reading its label at predicting where
the feature fires, mostly because a minority of labels are badly wrong and
the examples reveal it. On this pool, running experiments adds nothing
measurable to firing prediction beyond reading eight examples. The honest
headline is about labels and examples, not about tools.

## 16. Sentences unlike the examples: experiments still add nothing measurable

Run 2026-11-15, tagged `blind-5`. Design, the reasons for it and seven
predictions were committed before any pool was built (`results/DECISIONS.md`).

*Revised 2026-11-15 after a review of this section.* The first version
(commit f97c5e5) said that neither explanation beats the label, that the
hand-off between agents costs accuracy, and that no short explanation can
describe weak firings. The data does not support those three statements as
written: they depend on a scoring threshold, on a comparison that cannot
separate two causes, and on three tested descriptions. It also called six
random sentences per pool "new" that the explainers had already seen. What
follows is the corrected section. The primary result did not change.

**Question.** Section 15 found that reading eight top examples gives nearly
all of the gain, on a pool made of more top examples. Does an explanation
built with experiments do better on sentences the examples do not resemble?

**Setup.**

- *Features*: 24 of section 15's 25. One (11411) was refused because too few
  of its firings in ordinary text survive being cut to a 16-token window.
- *Explanations*: the ones section 15's agents wrote, reused unchanged.
- *Arms*: for each feature, three fresh agents with no tools predict the pool:
  from the label alone, from the label plus the examples-arm explanation, and
  from the label plus the investigator's explanation. Unlike sections 12 to
  15, the agent that predicts is not the agent that studied the feature.
- *Pool*, version 5: 4 held-out examples, 6 random sentences, and two new
  kinds (`scripts/scan_wild.py`). **Wild** (6): windows of ordinary text where
  the feature fires, sampled at random from its firings in about 18,000
  cached passages (2.3 million tokens in the whole cache; each feature's own
  examples and its neighbors' are left out). **Neighbor** (5): windows where
  one of the feature's five nearest features fires, chosen without looking
  at what the feature itself does there. 21 sentences; one feature (11243)
  has 3 held-out and 20.
- 72 agents, all sets complete, nothing contaminated. Graded locally with the
  same code as the MCP tool.

**Two scoring thresholds.** The truth counts as firing at 0.1 of max. A
prediction counts as "fires" at 2 on the 0 to 10 scale: that is the primary
definition in every run. A reader who correctly expects a firing at 0.15 of
max and writes 1.5 is therefore scored as predicting silence. That barely
mattered on top examples. It matters here, where most firings are weak, so
every F1 below is given at 2 (registered) and at 1 (matched to the truth
threshold, chosen after seeing the data).

**What the new sentences look like.** All 144 wild sentences fire, mostly
weakly: the median is 0.25 of max and 86 are below 0.3. The pool is half
strong and half weak by design, so that share understates how weak ordinary
firings are: across the scan, 82% of the passages where one of these
features fires have it below 0.3 of max. The feature fires on 26 of 120
neighbor sentences (22%) and 1 of 144 random ones.

**Results** (24 features, means; a metric that is undefined for a feature
is left out of its mean):

| sentences | metric | label | examples | investigated |
|---|---|---|---|---|
| wild, neighbor, random (registered) | F1, prediction of 2 or more | 0.66 | 0.59 | 0.64 |
| | F1, prediction of 1 or more | 0.68 | 0.71 | 0.70 |
| | Spearman | 0.57 | 0.63 | 0.69 |
| | peak_hit | 0.44 | 0.52 | 0.60 |
| held-out and random | F1, prediction of 2 or more | 0.87 | 0.84 | 0.83 |
| | F1, prediction of 1 or more | 0.83 | 0.92 | 0.90 |
| | Spearman | 0.83 | 0.89 | 0.90 |
| | peak_hit | 0.55 | 0.77 | 0.81 |

Paired contrasts (delta, wins / ties / losses, sign test):

| sentences | contrast | F1 at 2 | F1 at 1 | Spearman |
|---|---|---|---|---|
| wild, neighbor, random | **investigated minus examples (primary)** | **+0.05, 10 / 10 / 4, p = 0.18** | -0.01, 6 / 10 / 8, p = 0.79 | +0.03, 13 / 2 / 8, p = 0.38 |
| | investigated minus label | -0.02, 7 / 4 / 13, p = 0.26 | +0.02, 10 / 6 / 8, p = 0.81 | +0.09, 18 / 0 / 5, p = 0.011 |
| | examples minus label | -0.07, 6 / 5 / 13, p = 0.17 | +0.03, 12 / 5 / 7, p = 0.36 | +0.06, 16 / 2 / 6, p = 0.052 |
| held-out and random | investigated minus examples | -0.01, 5 / 13 / 6, p = 1 | -0.02, 4 / 14 / 6, p = 0.75 | +0.01, 8 / 9 / 7, p = 1 |
| | investigated minus label | -0.04, 2 / 14 / 8, p = 0.11 | +0.07, 12 / 10 / 2, p = 0.013 | +0.08, 15 / 3 / 6, p = 0.078 |
| | examples minus label | -0.03, 2 / 16 / 6, p = 0.29 | +0.09, 13 / 10 / 1, p = 0.002 | +0.07, 14 / 4 / 6, p = 0.12 |

Pooled rows, share predicted to fire:

| rows | n | at 2: label / examples / investigated | at 1: label / examples / investigated |
|---|---|---|---|
| wild, firing below 0.3 of max (recall) | 86 | 0.50 / 0.27 / 0.38 | 0.73 / 0.57 / 0.57 |
| wild, firing at 0.3 or more (recall) | 58 | 0.86 / 0.81 / 0.83 | 0.88 / 0.88 / 0.86 |
| held-out (recall) | 95 | 0.85 / 0.78 / 0.77 | 0.88 / 0.93 / 0.87 |
| neighbor, silent (false alarms) | 94 | 0.33 / 0.14 / 0.10 | 0.57 / 0.33 / 0.35 |
| random, silent (false alarms) | 143 | 0.04 / 0.01 / 0.01 | 0.15 / 0.04 / 0.02 |

**1. The primary comparison is not significant, and its small lean is a
threshold effect.** Experiments beat examples by +0.05 F1 at the registered
threshold (10 wins, 10 ties, 4 losses, p = 0.18). At the matched threshold
the difference is -0.01 with more losses than wins. The features that seemed
to show experiments helping (9261: 0.46 against 1.00; 312; 276) are level
or within 0.01 at the matched threshold (9261: 0.89 against 0.89). On the
weak firings that decide them, the examples arm's readers wrote 1 and the
investigator's readers wrote 2 to 4: both said "fires weakly". Two
runs, two kinds of pool, and no measurable gain from running experiments
over reading eight examples. With 24 features this rules out a large effect
and cannot rule out a small one.

**2. On the new sentences, explanations trade misses for false alarms, and
F1 does not move.** Both hold at either threshold. With an explanation, a
reader raises fewer false alarms on text where a neighboring feature fires
(0.33 falls to 0.14 and 0.10), which is the discrimination an explanation is
for. The same reader misses more of the weak firings in ordinary text (0.50
falls to 0.27 and 0.38). Net F1 is not significantly different from the bare
label at either threshold, in either direction.

**3. Explanations may improve the ranking.** Spearman on the registered
sentences rises from 0.57 to 0.63 and 0.69, and the investigator's gain over
the label has p = 0.011. On wild and neighbor sentences alone it is +0.16
with p = 0.052. One of several secondary tests: suggestive only.

**4. On sentences like the top examples, a written explanation does carry
over to another reader, at the matched threshold.** There both explanations
beat the label (+0.09, p = 0.002 and +0.07, p = 0.013); at the registered
threshold neither does. Two cautions. The threshold was chosen after the
fact. And half of these held-out sentences were in front of the agent who
wrote the explanation (see the next heading), which favors the explanation
arms here.

**5. Top examples can hide a second trigger.** Feature 13087's explanation
says the noun "payment", with tiers, and that is what its top examples show.
Of 20 firings sampled at random from those at 0.3 of max or more in
ordinary text, 14 are on forms of "measurement" and "probability" (0.30 to
0.38 of max), 2 on "notification" and 4 on "payment". Four of its six wild
sentences were of that kind, no arm predicted any of them, and all three
scored 0 on its new sentences. Neuronpedia's own list has " probability"
seven times from rank 32 down, at 0.29 to 0.44 of its maximum. A description written from top activations, or from
experiments designed around them, did not find it. Whether one written with
ordinary-text firings in view would was not tested.

**What this run cannot say about the hand-off.** With the same agent studying
and predicting, section 15 scored 0.81, 0.88 and 0.89 on held-out and random
sentences (these 24 features). Here separate readers scored 0.87, 0.84 and
0.83 at the registered threshold and 0.83, 0.92 and 0.90 at the matched one.
The label arm involves no hand-off and moved as much as the others, and the
pool changed between the runs. No cost of the hand-off can be read from this.

**Defects in this run, found in review.**

- *Shared sentences.* The pools were built with section 15's seed. All 144
  random sentences, 49 of the 95 held-out ones and one neighbor sentence
  were in the pool that section 15's agents had in their prompt when they
  wrote the explanations, and a few explanations name them. Only the wild
  and neighbor sentences are new to the explanations. On those alone the
  primary is unchanged (+0.05, 10 / 10 / 4, p = 0.18; -0.01 at the matched
  threshold) and F1 is 0.68, 0.60, 0.65 (0.72, 0.72, 0.71 matched). The
  overlap favors the explanation arms on held-out and random sentences.
- *Copied text.* 2 of the 144 wild sentences are text that also appears in
  one of the feature's own unshown examples or a neighbor's example, in a
  different passage. No shown example leaked.
- *Fixed order.* In every run so far, pools with the same seed and sizes put
  each source at the same ids. No agent has seen two pools, so nothing
  leaked, but it would in an interactive session that grades two features.

All three are fixed in the code for future runs (pool version 6, the scan,
and a check in `scripts/prepare_simulation.py`). This run was not repeated.

**Without the `<bos>` features.** Pool version 6 refuses features whose shown
examples mostly peak on `<bos>`: 7957, 11243 and also 2788. Without those
three (21 features) the primary is +0.07 (8 / 10 / 3, p = 0.23) at the
registered threshold and +0.01 at the matched one; investigated minus label
+0.01; examples minus label -0.06. Same conclusions.

**Earlier runs at the matched threshold** (exploratory). Investigated minus
label: section 14 +0.13 (15 / 5 / 3, p = 0.008); section 15 +0.11
(20 / 1 / 4, p = 0.002), against +0.08 and p = 0.077 at the registered
threshold. Investigated minus examples in section 15: +0.04 (10 / 12 / 3,
p = 0.09). The earlier conclusions hold at both thresholds.

**Predictions scored.** Right: the feature fires on about a quarter of
neighbor sentences (22%); the primary contrast is +0.04 to +0.08 with more
wins than losses and not significant (+0.05, p = 0.18); the label makes the
most false alarms on neighbor sentences and the investigator the fewest
(0.33, 0.10). Wrong: pool version 5 would refuse the two `<bos>` features
(it refused neither); F1 on new sentences 0.55, 0.62, 0.68 (0.66, 0.59,
0.64); recall on strong wild sentences would rise with each arm (flat at
0.81 to 0.86). Low: Spearman, investigated minus examples (+0.03 against
+0.05 to +0.10). Mixed: F1 on held-out and random sentences of 0.80, 0.87,
0.88 was wrong at the registered threshold (0.87, 0.84, 0.83) and close at
the matched one (0.83, 0.92, 0.90).

**Limits.** 24 features, one SAE, one layer. One model family writes and
reads the explanations. The explanations were written with section 15's pool
in view and capped at 120 words. The wild text is other features' example
passages, not a uniform sample. The agent that studied a feature never
predicted a wild or neighbor sentence, so whether it would do better than its
written explanation there is untested.

**Where sections 12 to 16 leave the question.**

- When the agent that studies a feature also predicts, it beats the label on
  held-out examples (48 features, +0.15 F1, p = 0.0009). The gain comes from
  the minority of labels that are badly wrong.
- Reading eight examples gets nearly all of that. Running experiments on top
  has not produced a measurable gain in firing prediction, on held-out
  examples or on ordinary text.
- On ordinary text and on neighboring features' text, a written explanation
  read by another agent gives fewer false alarms, more missed weak firings,
  possibly better ranking, and no change in F1.
- Most of what a feature does in ordinary text is weak, and for at least one
  feature the top examples hid a second trigger. Sampling where a feature
  fires in ordinary text is a cheap check that none of the arms had.
