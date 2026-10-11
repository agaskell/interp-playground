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
