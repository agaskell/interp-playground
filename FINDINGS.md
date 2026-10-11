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
