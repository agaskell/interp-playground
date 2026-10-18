# Agent Explanation Eval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two MCP tools, `eval_sentences` and `grade_explanation`, that let an agent predict an SAE feature's activations on a held-out sentence pool and get scored, logging label-vs-investigated comparisons to `results/explanation_evals.jsonl`.

**Architecture:** Pure functions in a new `evals.py` (pool assembly, prediction matching, scoring) are unit-tested without the model. `Backend` gains the two methods plus a process-wide record of prompts seen by other tools (contamination check) and a `_peak` helper that measures one sentence. `server.py` exposes the tools with protocol docstrings. A report script summarises the JSONL log.

**Tech Stack:** Python 3.12 (`statistics.correlation(method="ranked")` for Spearman, no new deps), nnsight 0.7, sae-lens, MCP Python SDK 2.x, pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-explanation-eval-design.md`

## Global Constraints

- Run everything with `uv run ...` from the repo root (nix flake + uv env; direnv may or may not be active in the executor's shell, so prefix `nix develop --command` if `uv` is missing).
- No new dependencies. Spearman uses `statistics.correlation(x, y, method="ranked")` (Python 3.12).
- Unit tests must not load the model; everything in `tests/` runs in under a second.
- All model access goes through `Backend._lock`; network calls (Neuronpedia) stay outside the lock.
- Position 0 is `<bos>`; always encode `resid[0, 1:]`.
- Constants from the spec, verbatim: `EXAMPLES_SHOWN = 8`, `FIRE_REL = 0.1`, `PREDICT_FIRE = 2`, held-out window 12 tokens before the peak and 4 after, defaults `n_heldout=4`, `n_random=8`, caps `n_heldout` 0-8, `n_random` 0-16, at most 4 prediction sets, at most 4 agent sentences per set.
- Commit after every task. Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>`. A gitleaks pre-commit hook runs locally; never pipe it through `tail`/`grep`.
- The running MCP server in Andy's Claude session does not pick up code changes until he runs `/mcp` and reconnects. Task 7 depends on that.

## Review Focus

1. Neuronpedia unreachable or a feature with few examples: `heldout` may be empty and `random` short. Grading must still work and report `n` honestly; Spearman on an empty or constant set is `None`, not a crash. (Task 2: `test_spearman_none_on_constant_or_empty`; Task 3: `test_build_pool_with_empty_sources`.)
2. Duplicate or conflicting predictions: the same pool id twice, or a prediction with neither id nor text. Must raise a `ValueError` naming the problem, never silently keep one. (Task 2: `test_match_predictions_rejects_duplicates_and_blanks`.)
3. `expected` outside 0-10 or given as a string: numeric strings are coerced, out-of-range values are clamped, non-numeric raises `ValueError`. (Task 2: `test_match_predictions_coerces_expected`.)
4. An agent sentence whose text equals a pool sentence (agent re-typed it without the id): it maps to that pool id instead of being counted twice as `agent`. (Task 2: `test_agent_text_matching_pool_maps_to_pool_id`.)
5. Neuronpedia token strings contain `▁` for spaces, `\n`, and sometimes `<bos>`: `example_text` must return clean prose or the model sees literal `▁` characters. (Task 1: `test_example_text_cleans_tokens`.)

---

### Task 1: Neuronpedia examples as plain text

**Files:**
- Modify: `src/interp_playground/neuronpedia.py:71-95` (`fetch_examples`)
- Modify: `docs/superpowers/specs/2026-10-18-explanation-eval-design.md` (one paragraph; see step 6)
- Test: `tests/test_neuronpedia.py` (create)

**Interfaces:**
- Produces: `load_examples(layer: int, feature: int) -> list[dict] | None` returning every cached example as `{"max": float, "tokens": list[str], "peak": int}`; `example_text(example: dict, before: int = 12, after: int = 4) -> str` returning a clean text window; `fetch_examples` unchanged in signature and output.

Note: the spec says `fetch_examples` caps the cache at 8. It does not; it caches every example Neuronpedia returns and slices on read. Only the split into `load_examples` + formatting is needed.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_neuronpedia.py
from interp_playground.neuronpedia import example_text


def test_example_text_cleans_tokens():
    ex = {
        "max": 50.0,
        "tokens": ["<bos>", "▁We", "▁drove", "▁across", "▁the", "▁old", "▁bridge", ".", "\n", "▁Next"],
        "peak": 6,
    }
    assert example_text(ex) == "We drove across the old bridge. Next"


def test_example_text_window_is_before_and_after_peak():
    toks = [f"▁t{i}" for i in range(30)]
    ex = {"max": 1.0, "tokens": toks, "peak": 20}
    text = example_text(ex, before=2, after=1)
    assert text == "t18 t19 t20 t21"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_neuronpedia.py -v`
Expected: FAIL with `ImportError: cannot import name 'example_text'`

- [ ] **Step 3: Split `fetch_examples` and add `example_text`**

Replace the whole `fetch_examples` function (lines 71-95) with:

```python
def load_examples(layer: int, feature: int) -> list[dict] | None:
    """All of Neuronpedia's top activating examples for a feature, cached on disk.

    Each example: {"max": peak activation, "tokens": token strings as
    Neuronpedia shows them (▁ for a leading space), "peak": index of the
    peak token}. Returns None if Neuronpedia is unreachable.
    """
    path = CACHE_DIR / "features" / sae_id(layer) / f"{feature}.json"
    if not path.exists():
        url = f"https://www.neuronpedia.org/api/feature/{MODEL_ID}/{sae_id(layer)}/{feature}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                raw = json.loads(resp.read())
        except (OSError, ValueError):
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([
            {"max": a["maxValue"], "tokens": a["tokens"], "peak": a["maxValueTokenIndex"]}
            for a in raw.get("activations", [])
        ]))
    return json.loads(path.read_text())


def example_text(example: dict, before: int = 12, after: int = 4) -> str:
    """A plain-text window around an example's peak token, for running through the model."""
    toks, i = example["tokens"], example["peak"]
    window = [t for t in toks[max(0, i - before):i + 1 + after] if t != "<bos>"]
    return " ".join("".join(window).replace("▁", " ").split())


def fetch_examples(layer: int, feature: int, n: int = 8, context: int = 12) -> list[dict] | None:
    """Top activating examples for display, with the peak token in [[brackets]].

    Returns None if Neuronpedia is unreachable.
    """
    examples = load_examples(layer, feature)
    if examples is None:
        return None
    out = []
    for a in examples[:n]:
        toks, i = a["tokens"], a["peak"]
        snippet = "".join(toks[max(0, i - context):i]) + f"[[{toks[i]}]]" + "".join(toks[i + 1:i + 4])
        out.append({"max": round(a["max"], 1), "text": snippet.replace("\n", "\\n")})
    return out
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest -q`
Expected: 17 passed (15 existing + 2 new)

- [ ] **Step 5: Check the display path still works without the network**

Run: `uv run python -c "from interp_playground.neuronpedia import fetch_examples, load_examples, example_text; ex = load_examples(12, 7272); print(len(ex)); print(example_text(ex[8])); print(fetch_examples(12, 7272, n=1))"`
Expected: a count above 8 (the cache from earlier `feature_info` calls), a clean sentence containing "bridge", and one bracketed snippet. If the count is 8 or fewer, Neuronpedia returned few examples for this feature; note it and continue.

- [ ] **Step 6: Correct the spec**

In `docs/superpowers/specs/2026-10-18-explanation-eval-design.md`, replace the paragraph beginning "`fetch_examples` currently caps at 8 examples in its cache" with:

```
`fetch_examples` already caches every example Neuronpedia returns and
slices on read; it is split into `load_examples` (raw cached list) and the
display formatting, plus `example_text` for the plain-text window.
```

and in "Backend changes" replace "`neuronpedia.fetch_examples` caches all examples; a new `example_text(example)` returns the plain-text window." with "`neuronpedia.load_examples` returns the raw cached list; `example_text(example)` returns the plain-text window."

- [ ] **Step 7: Commit**

```bash
git add src/interp_playground/neuronpedia.py tests/test_neuronpedia.py docs/superpowers/specs/2026-10-18-explanation-eval-design.md
git commit -m "neuronpedia: load_examples and example_text for eval sentences

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 2: Scoring and prediction matching (pure)

**Files:**
- Create: `src/interp_playground/evals.py`
- Test: `tests/test_evals.py` (create)

**Interfaces:**
- Produces, in `interp_playground.evals`:
  - constants `EXAMPLES_SHOWN = 8`, `FIRE_REL = 0.1`, `PREDICT_FIRE = 2`, `MAX_EXPECTED = 10`, `MAX_AGENT = 4`, `MAX_SETS = 4`
  - `match_predictions(pool: list[dict], predictions: list[dict]) -> list[dict]`: pool items are `{"id": int, "text": str, "source": str}`; predictions are `{"id": int | None, "text": str | None, "expected": number, "on": str | None}`; returns rows `{"id", "text", "source", "expected" (float, clamped), "on"}`.
  - `score_set(rows: list[dict]) -> dict`: rows must additionally carry `"rel": float` and `"peak_token": str` (added by the backend); returns `{"n", "spearman", "precision", "recall", "f1", "peak_hit", "by_source": {source: {"n", "precision", "recall", "f1"}}, "rows"}` and adds `fired`, `predicted_fired`, `peak_hit` to each row.
  - `spearman(expected: list[float], rel: list[float]) -> float | None`, `peak_matches(on: str | None, peak_token: str) -> bool`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_evals.py
import pytest

from interp_playground.evals import (
    FIRE_REL,
    match_predictions,
    peak_matches,
    score_set,
    spearman,
)

POOL = [
    {"id": 0, "text": "We drove across the old bridge.", "source": "heldout"},
    {"id": 1, "text": "The soup needs more salt.", "source": "random"},
    {"id": 2, "text": "Music can bridge divides.", "source": "heldout"},
]


def test_spearman_ranks_and_clamps_weak_activations():
    # Perfect ranking once sub-threshold rels are treated as 0.
    assert spearman([0, 5, 10], [0.0, 0.5, 1.0]) == 1.0
    # Ranking among non-firing sentences must not count: 0.02 vs 0.05 are both 0.
    assert spearman([0, 1, 10], [0.05, 0.02, 1.0]) == spearman([1, 0, 10], [0.05, 0.02, 1.0])


def test_spearman_none_on_constant_or_empty():
    assert spearman([0, 0, 0], [0.0, 0.5, 1.0]) is None
    assert spearman([], []) is None
    assert spearman([3], [0.5]) is None


def test_peak_matches_is_lenient_about_spacing_case_and_plurals():
    assert peak_matches("bridge", " bridges")
    assert peak_matches(" Bridges", "bridge")
    assert peak_matches("San Francisco", " San")
    assert not peak_matches("river", " bridge")
    assert not peak_matches(None, " bridge")
    assert not peak_matches("", " bridge")


def test_match_predictions_by_id_and_agent_text():
    preds = [
        {"id": 0, "expected": 9, "on": "bridge"},
        {"id": 1, "expected": 0},
        {"id": 2, "expected": 4, "on": "bridge"},
        {"text": "The dentist fitted a bridge.", "expected": 7, "on": "bridge"},
    ]
    rows = match_predictions(POOL, preds)
    assert [r["source"] for r in rows] == ["heldout", "random", "heldout", "agent"]
    assert rows[3]["id"] is None and rows[3]["text"] == "The dentist fitted a bridge."
    assert rows[1]["on"] is None


def test_match_predictions_requires_every_pool_id():
    with pytest.raises(ValueError, match=r"missing.*\[1, 2\]"):
        match_predictions(POOL, [{"id": 0, "expected": 9}])


def test_match_predictions_rejects_duplicates_and_blanks():
    with pytest.raises(ValueError, match="twice"):
        match_predictions(POOL, [{"id": 0, "expected": 9}, {"id": 0, "expected": 1},
                                 {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    with pytest.raises(ValueError, match="id .* or text"):
        match_predictions(POOL, [{"expected": 9}, {"id": 0, "expected": 9},
                                 {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    with pytest.raises(ValueError, match="unknown sentence id 7"):
        match_predictions(POOL, [{"id": 7, "expected": 9}, {"id": 0, "expected": 9},
                                 {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])


def test_match_predictions_coerces_expected():
    rows = match_predictions(POOL, [{"id": 0, "expected": "7"}, {"id": 1, "expected": -3},
                                    {"id": 2, "expected": 15}])
    assert [r["expected"] for r in rows] == [7.0, 0.0, 10.0]
    with pytest.raises(ValueError, match="expected"):
        match_predictions(POOL, [{"id": 0, "expected": "high"}, {"id": 1, "expected": 0},
                                 {"id": 2, "expected": 0}])


def test_agent_text_matching_pool_maps_to_pool_id():
    rows = match_predictions(POOL, [{"text": "We drove across the old bridge.", "expected": 9},
                                    {"id": 1, "expected": 0}, {"id": 2, "expected": 0}])
    assert rows[0]["id"] == 0 and rows[0]["source"] == "heldout"


def test_match_predictions_caps_agent_sentences():
    extra = [{"text": f"agent sentence {i}", "expected": 0} for i in range(5)]
    with pytest.raises(ValueError, match="at most 4 agent"):
        match_predictions(POOL, [{"id": 0, "expected": 0}, {"id": 1, "expected": 0},
                                 {"id": 2, "expected": 0}, *extra])


def test_score_set_metrics():
    rows = [
        # fired, predicted, peak hit
        {"id": 0, "text": "a", "source": "heldout", "expected": 9.0, "on": "bridge", "rel": 1.0, "peak_token": " bridge"},
        # fired, predicted, peak miss
        {"id": 2, "text": "b", "source": "heldout", "expected": 4.0, "on": "music", "rel": 0.3, "peak_token": " bridge"},
        # not fired, not predicted
        {"id": 1, "text": "c", "source": "random", "expected": 0.0, "on": None, "rel": 0.0, "peak_token": " salt"},
        # not fired (weak), predicted: false positive
        {"id": None, "text": "d", "source": "agent", "expected": 5.0, "on": "gap", "rel": 0.05, "peak_token": " gap"},
        # fired, not predicted: false negative, no `on` so excluded from peak_hit
        {"id": None, "text": "e", "source": "agent", "expected": 1.0, "on": None, "rel": 0.2, "peak_token": " span"},
    ]
    s = score_set(rows)
    assert s["n"] == 5
    assert (s["precision"], s["recall"]) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["peak_hit"] == 0.5
    assert s["by_source"]["heldout"] == {"n": 2, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert s["by_source"]["random"] == {"n": 1, "precision": None, "recall": None, "f1": None}
    assert s["by_source"]["agent"]["precision"] == 0.0 and s["by_source"]["agent"]["recall"] == 0.0
    assert s["by_source"]["agent"]["f1"] == 0.0
    assert rows[3]["fired"] is False and rows[3]["predicted_fired"] is True
    assert rows[3]["peak_hit"] is None  # not fired, so no peak to hit
    assert rows[0]["peak_hit"] is True and rows[1]["peak_hit"] is False
    assert s["spearman"] is not None and FIRE_REL == 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_evals.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'interp_playground.evals'`

- [ ] **Step 3: Write `evals.py`**

```python
# src/interp_playground/evals.py
"""Pure pieces of the agent explanation eval: sentence pool, matching, scoring.

The backend supplies texts and ground-truth activations; nothing here
touches the model or the network, so it all runs in unit tests.
"""

import random
import statistics
from typing import Any

# feature_info shows this many top examples; the rest are held out for the eval.
EXAMPLES_SHOWN = 8
# Actual activation / max_act at or above this counts as firing.
FIRE_REL = 0.1
# A prediction (0-10) at or above this predicts firing.
PREDICT_FIRE = 2
MAX_EXPECTED = 10
MAX_AGENT = 4
MAX_SETS = 4
SOURCES = ("heldout", "random", "agent")


def pick_random_features(
    n_features: int, hidden: list[bool], exclude: int, seed: int, n: int
) -> list[int]:
    """`n` distinct feature ids that are not hidden and not `exclude`, chosen by seed."""
    candidates = [i for i in range(n_features) if not hidden[i] and i != exclude]
    return random.Random(seed).sample(candidates, min(n, len(candidates)))


def build_pool(heldout: list[str], random_texts: list[str], seed: int) -> list[dict[str, Any]]:
    """Shuffle the sources together; ids are 0..n-1 in shuffled order."""
    items = [(t, "heldout") for t in heldout] + [(t, "random") for t in random_texts]
    random.Random(seed).shuffle(items)
    return [{"id": i, "text": t, "source": s} for i, (t, s) in enumerate(items)]


def match_predictions(pool: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair predictions with pool sentences by id; anything else is an agent sentence.

    Every pool id must be predicted exactly once. `expected` is coerced to a
    float and clamped to 0-10.
    """
    by_id = {p["id"]: p for p in pool}
    by_text = {p["text"]: p for p in pool}
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for pred in predictions:
        pid, text = pred.get("id"), (pred.get("text") or "").strip()
        if pid is None and text in by_text:
            pid = by_text[text]["id"]
        if pid is not None:
            if pid not in by_id:
                raise ValueError(f"unknown sentence id {pid}")
            if pid in seen:
                raise ValueError(f"sentence id {pid} predicted twice")
            seen.add(pid)
            base = by_id[pid]
        elif text:
            base = {"id": None, "text": text, "source": "agent"}
        else:
            raise ValueError("each prediction needs an id (pool sentence) or text (agent sentence)")
        try:
            expected = float(pred.get("expected", 0))
        except (TypeError, ValueError):
            raise ValueError(f"expected must be a number 0-10, got {pred.get('expected')!r}") from None
        rows.append({
            **base,
            "expected": min(max(expected, 0.0), float(MAX_EXPECTED)),
            "on": pred.get("on") or None,
        })
    missing = sorted(set(by_id) - seen)
    if missing:
        raise ValueError(f"predictions missing for sentence ids {missing}")
    n_agent = sum(r["source"] == "agent" for r in rows)
    if n_agent > MAX_AGENT:
        raise ValueError(f"at most {MAX_AGENT} agent sentences per set, got {n_agent}")
    return rows


def peak_matches(on: str | None, peak_token: str) -> bool:
    """Does the predicted peak word match the actual peak token? Lenient on case,
    spacing and inflection: "bridge" matches " bridges" and vice versa."""
    if not on:
        return False
    a = on.strip().lstrip("▁").lower()
    b = peak_token.strip().lstrip("▁").lower()
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def spearman(expected: list[float], rel: list[float]) -> float | None:
    """Rank correlation, with sub-threshold activations clamped to 0 first so
    ordering among non-firing sentences doesn't count. None if undefined."""
    ys = [0.0 if r < FIRE_REL else r for r in rel]
    try:
        return round(statistics.correlation(list(map(float, expected)), ys, method="ranked"), 3)
    except statistics.StatisticsError:
        return None


def _prf(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    tp = sum(r["fired"] and r["predicted_fired"] for r in rows)
    fp = sum((not r["fired"]) and r["predicted_fired"] for r in rows)
    fn = sum(r["fired"] and (not r["predicted_fired"]) for r in rows)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    if p is None or rc is None:
        f1 = None
    else:
        f1 = 2 * p * rc / (p + rc) if p + rc else 0.0
    rnd = lambda v: None if v is None else round(v, 3)
    return {"precision": rnd(p), "recall": rnd(rc), "f1": rnd(f1)}


def score_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one prediction set. Rows need expected, rel, peak_token, on, source;
    fired / predicted_fired / peak_hit are added to each row in place."""
    for r in rows:
        r["fired"] = r["rel"] >= FIRE_REL
        r["predicted_fired"] = r["expected"] >= PREDICT_FIRE
        r["peak_hit"] = peak_matches(r.get("on"), r["peak_token"]) if r["fired"] else None
    hits = [r["peak_hit"] for r in rows if r["fired"] and r.get("on")]
    by_source = {}
    for source in SOURCES:
        group = [r for r in rows if r["source"] == source]
        if group:
            by_source[source] = {"n": len(group), **_prf(group)}
    return {
        "n": len(rows),
        "spearman": spearman([r["expected"] for r in rows], [r["rel"] for r in rows]),
        **_prf(rows),
        "peak_hit": round(sum(hits) / len(hits), 3) if hits else None,
        "by_source": by_source,
        "rows": rows,
    }
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_evals.py -v`
Expected: 10 passed. If `test_spearman_ranks_and_clamps_weak_activations` fails on the first assertion with a value like 0.999, change the assertion to `== pytest.approx(1.0)`.

- [ ] **Step 5: Commit**

```bash
git add src/interp_playground/evals.py tests/test_evals.py
git commit -m "evals: prediction matching and scoring for the explanation eval

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 3: Sentence pool (pure)

**Files:**
- Modify: `src/interp_playground/evals.py` (functions already written in Task 2: `pick_random_features`, `build_pool`)
- Test: `tests/test_evals.py` (append)

**Interfaces:**
- Produces: `pick_random_features(n_features, hidden, exclude, seed, n) -> list[int]` and `build_pool(heldout, random_texts, seed) -> list[{"id","text","source"}]`, deterministic in `seed`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_evals.py`:

```python
from interp_playground.evals import build_pool, pick_random_features


def test_pick_random_features_skips_hidden_and_self_and_is_seeded():
    hidden = [False] * 20
    hidden[3] = hidden[4] = True
    picked = pick_random_features(20, hidden, exclude=7, seed=0, n=10)
    assert len(picked) == len(set(picked)) == 10
    assert not {3, 4, 7} & set(picked)
    assert picked == pick_random_features(20, hidden, exclude=7, seed=0, n=10)
    assert picked != pick_random_features(20, hidden, exclude=7, seed=1, n=10)
    # Asking for more than exist returns what exists.
    assert len(pick_random_features(5, [False] * 5, exclude=0, seed=0, n=10)) == 4


def test_build_pool_shuffles_deterministically_and_tags_sources():
    pool = build_pool(["h1", "h2"], ["r1", "r2", "r3"], seed=0)
    assert [p["id"] for p in pool] == [0, 1, 2, 3, 4]
    assert sorted(p["text"] for p in pool) == ["h1", "h2", "r1", "r2", "r3"]
    assert {p["text"]: p["source"] for p in pool} == {
        "h1": "heldout", "h2": "heldout", "r1": "random", "r2": "random", "r3": "random"
    }
    assert pool == build_pool(["h1", "h2"], ["r1", "r2", "r3"], seed=0)
    assert [p["text"] for p in pool] != [p["text"] for p in build_pool(["h1", "h2"], ["r1", "r2", "r3"], seed=1)]


def test_build_pool_with_empty_sources():
    assert build_pool([], [], seed=0) == []
    pool = build_pool([], ["r1"], seed=0)
    assert pool == [{"id": 0, "text": "r1", "source": "random"}]
    # Grading an empty pool with only agent sentences still scores.
    rows = match_predictions([], [{"text": "x", "expected": 3}])
    rows[0].update({"rel": 0.0, "peak_token": " x"})
    s = score_set(rows)
    # One false positive: precision 0, recall undefined (nothing fired), so f1 is None.
    assert s["n"] == 1 and s["spearman"] is None
    assert s["precision"] == 0.0 and s["recall"] is None and s["f1"] is None
```

- [ ] **Step 2: Run tests**

Run: `uv run pytest tests/test_evals.py -v`
Expected: 13 passed (the functions exist from Task 2). If `test_build_pool_shuffles_deterministically_and_tags_sources` fails on the seed-1 inequality because both shuffles happen to coincide, change `seed=1` to `seed=2`.

- [ ] **Step 3: Commit**

```bash
git add tests/test_evals.py
git commit -m "evals: tests for pool construction

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 4: Backend methods, contamination record, results log

**Files:**
- Modify: `src/interp_playground/backend.py` (imports at top; `__init__` lines 51-62; `run_prompt`, `steer`, `logit_lens`, `ablate` entry points; new methods after `feature_info`)

**Interfaces:**
- Consumes: `neuronpedia.load_examples`, `neuronpedia.example_text` (Task 1); `evals.pick_random_features`, `build_pool`, `match_predictions`, `score_set`, `EXAMPLES_SHOWN`, `MAX_SETS` (Tasks 2-3).
- Produces: `Backend.eval_sentences(feature: int, layer: int, seed: int, n_heldout: int, n_random: int) -> dict` returning `{"feature", "layer", "seed", "n_heldout", "n_random", "sentences": [{"id", "text"}]}`; `Backend.grade_explanation(feature: int, layer: int, explanation: str, predictions: dict[str, list[dict]], seed: int, n_heldout: int, n_random: int) -> dict` returning the record described in the spec; module constant `RESULTS_PATH: Path`.

- [ ] **Step 1: Add imports and the results path**

At the top of `backend.py`, after `import threading`, add:

```python
import json
import os
from datetime import datetime, timezone
from pathlib import Path
```

After the `from interp_playground import neuronpedia` line add:

```python
from interp_playground.evals import (
    EXAMPLES_SHOWN,
    MAX_SETS,
    build_pool,
    match_predictions,
    pick_random_features,
    score_set,
)
```

After the `ARTIFACT_PROBES = [...]` list add:

```python
# Where grade_explanation appends one JSON line per call. Override with the
# INTERP_PLAYGROUND_RESULTS env var (the smoke test does, to keep the repo's
# log clean).
RESULTS_PATH = Path(
    os.environ.get(
        "INTERP_PLAYGROUND_RESULTS",
        Path(__file__).resolve().parents[2] / "results" / "explanation_evals.jsonl",
    )
)


def _normalize(text: str) -> str:
    return " ".join(text.split())
```

- [ ] **Step 2: Add the contamination record to `__init__` and the four tools**

In `__init__`, after `self._hidden: dict[int, dict[str, torch.Tensor]] = {}` add:

```python
        # Every prompt the other tools have run, for the eval's contamination
        # check. Never cleared: several features can be evaluated per session.
        self._seen_prompts: set[str] = set()
```

Add as the first statement of the bodies of `run_prompt`, `steer`, `logit_lens` and `ablate` (before any validation):

```python
        self._seen_prompts.add(_normalize(prompt))
```

- [ ] **Step 3: Add the eval methods**

Insert after `feature_info` (before `def logit_lens`):

```python
    def eval_sentences(
        self, feature: int, layer: int, seed: int, n_heldout: int, n_random: int
    ) -> dict[str, Any]:
        self._check(feature, layer)
        pool = self._pool(feature, layer, seed, n_heldout, n_random)
        return {
            "feature": feature,
            "layer": layer,
            "seed": seed,
            "n_heldout": n_heldout,
            "n_random": n_random,
            "sentences": [{"id": p["id"], "text": p["text"]} for p in pool],
        }

    def grade_explanation(
        self,
        feature: int,
        layer: int,
        explanation: str,
        predictions: dict[str, list[dict[str, Any]]],
        seed: int,
        n_heldout: int,
        n_random: int,
    ) -> dict[str, Any]:
        self._check(feature, layer)
        if not predictions:
            raise ValueError("predictions must contain at least one named set")
        if len(predictions) > MAX_SETS:
            raise ValueError(f"at most {MAX_SETS} prediction sets, got {len(predictions)}")
        pool = self._pool(feature, layer, seed, n_heldout, n_random)
        sets = {name: match_predictions(pool, preds) for name, preds in predictions.items()}

        texts = sorted({r["text"] for rows in sets.values() for r in rows})
        with self._lock:
            model = self._get_model()
            self._prepare_layer(layer)
            truth = {t: self._peak(model, feature, layer, t) for t in texts}
            label = self._tables[layer][feature].label
        for rows in sets.values():
            for r in rows:
                r.update(truth[r["text"]])

        record = {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "feature": feature,
            "layer": layer,
            "label": label,
            "explanation": explanation,
            "seed": seed,
            "n_heldout": n_heldout,
            "n_random": n_random,
            "contaminated": [p["id"] for p in pool if _normalize(p["text"]) in self._seen_prompts],
            "sets": {name: score_set(rows) for name, rows in sets.items()},
        }
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RESULTS_PATH.open("a") as f:
            f.write(json.dumps(record) + "\n")
        return record

    def _pool(
        self, feature: int, layer: int, seed: int, n_heldout: int, n_random: int
    ) -> list[dict[str, Any]]:
        """The eval's sentence pool: held-out own examples plus other features' text.

        Deterministic in (feature, layer, seed, n_heldout, n_random). Network
        reads (cached per feature) happen outside the model lock.
        """
        with self._lock:
            self._prepare_layer(layer)
            masks = self._hidden[layer]
            hidden = (masks["dense"] | masks["position_artifact"]).tolist()
        own = neuronpedia.load_examples(layer, feature) or []
        heldout = [neuronpedia.example_text(e) for e in own[EXAMPLES_SHOWN:EXAMPLES_SHOWN + n_heldout]]
        random_texts = []
        for other in pick_random_features(D_SAE, hidden, feature, seed, n_random):
            examples = neuronpedia.load_examples(layer, other)
            if examples:
                random_texts.append(neuronpedia.example_text(examples[0]))
        return build_pool([t for t in heldout if t], [t for t in random_texts if t], seed)

    def _peak(self, model: LanguageModel, feature: int, layer: int, text: str) -> dict[str, Any]:
        """Ground truth for one sentence: the feature's peak activation and where."""
        ids = model.tokenizer(text, return_tensors="pt")["input_ids"]
        if ids.shape[1] < 2:
            raise ValueError(f"sentence has no tokens: {text!r}")
        with torch.no_grad(), model.trace(ids):
            resid = model.model.layers[layer].output.save()
        with torch.no_grad():
            acts = self._saes[layer].encode(resid[0, 1:])[:, feature].cpu()
        i = int(acts.argmax())
        act = float(acts[i])
        return {
            "act": round(act, 2),
            "rel": round(act / float(self._max_act[layer][feature]), 3),
            "peak_token": model.tokenizer.decode(ids[0, 1 + i]),
        }
```

- [ ] **Step 4: Run the unit tests and a direct backend check**

Run: `uv run pytest -q`
Expected: 30 passed, nothing touches the backend.

Run (loads the model, ~10 s; writes to a scratch log, not the repo):

```bash
INTERP_PLAYGROUND_RESULTS=/private/tmp/claude-501/-Users-andy-projects-interp-playground/f665cfad-6916-442a-a496-6a68a7ac7e5e/scratchpad/evals_check.jsonl \
uv run python -c "
from interp_playground.backend import Backend
b = Backend()
pool = b.eval_sentences(7272, 12, seed=0, n_heldout=2, n_random=3)
print(pool)
preds = {'zeros': [{'id': s['id'], 'expected': 0} for s in pool['sentences']]
                  + [{'text': 'We drove across the old bridge.', 'expected': 9, 'on': 'bridge'}]}
r = b.grade_explanation(7272, 12, 'check', preds, seed=0, n_heldout=2, n_random=3)
s = r['sets']['zeros']
print({k: s[k] for k in ('n', 'precision', 'recall', 'f1', 'spearman', 'peak_hit')})
for row in s['rows']: print(row['source'], row['rel'], repr(row['peak_token']), row['text'][:50])
print('contaminated', r['contaminated'])
"
```

Expected: 5 pool sentences with ids 0-4 and no source tags; the agent sentence row shows `rel` about 1.0 on `' bridge'` with `peak_hit True`; held-out rows have `rel` well above 0.1 (they are the feature's own examples); random rows near 0; `recall` low (we predicted zeros) and `contaminated []`. Confirm the scratch file has one line: `wc -l` on the path above.

- [ ] **Step 5: Commit**

```bash
git add src/interp_playground/backend.py
git commit -m "backend: eval_sentences and grade_explanation with contamination record

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 5: MCP tools and smoke test

**Files:**
- Modify: `src/interp_playground/server.py` (new tools after `ablate`; `TOOLS` tuple)
- Modify: `scripts/mcp_smoke.py`

**Interfaces:**
- Consumes: `backend.eval_sentences`, `backend.grade_explanation` (Task 4).
- Produces: MCP tools `eval_sentences(feature, layer=12, seed=0, n_heldout=4, n_random=8)` and `grade_explanation(feature, explanation, predictions, layer=12, seed=0, n_heldout=4, n_random=8)`; `server.TOOLS` includes both.

- [ ] **Step 1: Add the tools**

Insert after the `ablate` tool (before `TOOLS = (...)`):

```python
@server.tool()
def eval_sentences(
    feature: int,
    layer: int = DEFAULT_LAYER,
    seed: int = 0,
    n_heldout: int = 4,
    n_random: int = 8,
) -> dict[str, Any]:
    """Start an explanation eval: get sentences to predict a feature's activations on.

    Protocol (do it in this order):
    1. Call this. You get sentences as text with ids, shuffled, and nothing
       else: some are the feature's own top examples that feature_info did
       not show you, some are random text from other features' examples.
    2. BEFORE using any other tool on this feature, write a prediction for
       every sentence from the Neuronpedia label alone. This is the baseline
       set, named "label".
    3. Investigate with feature_info, run_prompt (on sentences of your own
       choosing, not these), steer, ablate. Write an explanation and a second
       prediction set named "investigated" for the same sentences. Add up to
       4 sentences of your own (near-misses, edge cases) by giving `text`
       instead of `id`.
    4. Call grade_explanation once with both sets.

    Do not run these sentences through run_prompt or the other tools before
    grading: any that you do are reported as `contaminated`. A good score is
    necessary, not sufficient; check the investigated explanation with one
    causal probe (steer or ablate) and note the result in your write-up.

    Args:
        feature: Feature id (0-16383).
        layer: Residual-stream layer of the feature's SAE (0-25). Default 12.
        seed: Pool seed; pass the same seed and counts to grade_explanation.
        n_heldout: Held-out own examples to include (0-8). Default 4.
        n_random: Random-text sentences to include (0-16). Default 8.
    """
    return backend.eval_sentences(
        feature,
        layer,
        seed,
        n_heldout=max(0, min(n_heldout, 8)),
        n_random=max(0, min(n_random, 16)),
    )


@server.tool()
def grade_explanation(
    feature: int,
    explanation: str,
    predictions: dict[str, list[dict[str, Any]]],
    layer: int = DEFAULT_LAYER,
    seed: int = 0,
    n_heldout: int = 4,
    n_random: int = 8,
) -> dict[str, Any]:
    """Score prediction sets from eval_sentences against the feature's real activations.

    `predictions` maps a set name ("label", "investigated") to a list of
    {"id": <sentence id>, "expected": 0-10, "on": "<word>"} for every pool
    sentence, plus optional {"text": "<your sentence>", "expected", "on"}
    entries (at most 4 per set). `expected` is the predicted peak activation
    relative to the feature's max: 10 = as strong as a top example, 0 = does
    not fire. `on` (optional) is the word you expect the peak on.

    Pass the same seed, n_heldout and n_random as eval_sentences.

    Per set, returns: spearman (rank correlation of expected vs actual,
    sub-threshold activations counted as 0), precision/recall/f1 for firing
    (actual rel >= 0.1 vs expected >= 2), the same by source (heldout /
    random / agent), peak_hit (fraction of true firings whose peak token you
    named), and per-sentence rows with the truth. Also returns `contaminated`:
    pool ids that were run through other tools before grading. Every call is
    appended to results/explanation_evals.jsonl.

    Read the rows, not just the scores: a wrong `on` with the right `expected`
    usually means the explanation names the topic but not the trigger.
    """
    return backend.grade_explanation(
        feature,
        layer,
        explanation,
        predictions,
        seed,
        n_heldout=max(0, min(n_heldout, 8)),
        n_random=max(0, min(n_random, 16)),
    )
```

Change `TOOLS` to:

```python
TOOLS = (
    "run_prompt", "steer", "feature_info", "logit_lens", "ablate",
    "eval_sentences", "grade_explanation",
)
```

- [ ] **Step 2: Extend the smoke test**

In `scripts/mcp_smoke.py`, add `import os` with the other imports, change the server parameters to:

```python
    params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", str(ROOT), "interp-playground-mcp"],
        env={**os.environ, "INTERP_PLAYGROUND_RESULTS": str(ROOT / ".smoke_evals.jsonl")},
    )
```

and append inside `main()` after the feature_info/logit_lens/ablate loop:

```python
        result = await client.call_tool(
            "eval_sentences", {"feature": 7272, "n_heldout": 2, "n_random": 3}
        )
        if result.is_error:
            sys.exit(f"eval_sentences error: {result.content}")
        pool = result.structured_content["sentences"]
        assert all(set(s) == {"id", "text"} for s in pool), "pool leaks fields"
        print("eval_sentences:", len(pool), "sentences")

        preds = {"zeros": [{"id": s["id"], "expected": 0} for s in pool]
                 + [{"text": "We drove across the old bridge.", "expected": 9, "on": "bridge"}]}
        result = await client.call_tool(
            "grade_explanation",
            {"feature": 7272, "explanation": "smoke", "predictions": preds,
             "n_heldout": 2, "n_random": 3},
        )
        if result.is_error:
            sys.exit(f"grade_explanation error: {result.content}")
        s = result.structured_content["sets"]["zeros"]
        assert s["n"] == len(pool) + 1
        agent_row = next(r for r in s["rows"] if r["source"] == "agent")
        assert agent_row["peak_hit"] is True, agent_row
        print("grade_explanation:", {k: s[k] for k in ("n", "f1", "spearman", "peak_hit")})
```

Add `.smoke_evals.jsonl` to `.gitignore`.

- [ ] **Step 3: Run the smoke test and unit tests**

Run: `uv run python scripts/mcp_smoke.py 2>&1 | grep -v "Loading weights"; echo "exit=${PIPESTATUS[0]}"`
Expected: the earlier lines unchanged, then `eval_sentences: 5 sentences` and a `grade_explanation:` line; `exit=0`. The tools list line should now show seven tools.

Run: `uv run pytest -q`
Expected: 30 passed.

- [ ] **Step 4: Commit**

```bash
git add src/interp_playground/server.py scripts/mcp_smoke.py .gitignore
git commit -m "server: eval_sentences and grade_explanation tools

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
```

---

### Task 6: Report script and docs

**Files:**
- Create: `scripts/eval_report.py`
- Modify: `README.md` (MCP server section after the `logit_lens` paragraph; Scripts section; Roadmap bullet)

**Interfaces:**
- Consumes: the JSONL record format from Task 4.

- [ ] **Step 1: Write the report script**

```python
# scripts/eval_report.py
"""Summarise explanation-eval results: one line per prediction set, with the
label-vs-investigated F1 difference per run.

Usage: uv run python scripts/eval_report.py [results/explanation_evals.jsonl]
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fmt(x: float | None) -> str:
    return "    -" if x is None else f"{x:5.2f}"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "explanation_evals.jsonl"
    if not path.exists():
        sys.exit(f"no results at {path}")
    print(f"{'time':19} {'feat':>5} {'set':13} {'n':>3} {'f1':>5} {'prec':>5} {'rec':>5} "
          f"{'rho':>5} {'peak':>5} {'cont':>4}  label")
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        for name, s in rec["sets"].items():
            print(f"{rec['time'][:19]:19} {rec['feature']:>5} {name[:13]:13} {s['n']:>3} "
                  f"{fmt(s['f1'])} {fmt(s['precision'])} {fmt(s['recall'])} "
                  f"{fmt(s['spearman'])} {fmt(s['peak_hit'])} {len(rec['contaminated']):>4}  "
                  f"{(rec['label'] or '')[:40]}")
        sets = rec["sets"]
        if "label" in sets and "investigated" in sets:
            a, b = sets["label"]["f1"], sets["investigated"]["f1"]
            delta = None if a is None or b is None else b - a
            print(f"{'':19} {'':>5} {'delta f1':13} {'':>3} {fmt(delta)}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Check it against the scratch log from Task 4**

Run: `uv run python scripts/eval_report.py /private/tmp/claude-501/-Users-andy-projects-interp-playground/f665cfad-6916-442a-a496-6a68a7ac7e5e/scratchpad/evals_check.jsonl`
Expected: a header and one `zeros` line for feature 7272. No `delta f1` line, since there is no label/investigated pair.

- [ ] **Step 3: README**

After the paragraph in "MCP server" that ends "`logit_lens` shows what each layer would predict at a position, and when the final answer forms." add:

```
`eval_sentences` and `grade_explanation` run an explanation eval: the agent
predicts a feature's activations on a mixed pool (held-out Neuronpedia
examples, random text from other features, its own near-misses), first from
the label alone and again after investigating, and the server scores both
sets with Spearman correlation, firing F1 by source, and peak-token hits.
Results append to `results/explanation_evals.jsonl`; `scripts/eval_report.py`
prints the label-vs-investigated table. Design:
`docs/superpowers/specs/2026-10-18-explanation-eval-design.md`.
```

In "Scripts", after the `scripts/tool.py` line add:

```
uv run python scripts/eval_report.py   # label vs investigated scores per feature
```

In "Roadmap", replace the bullet beginning "- **Agent explanation evals.**" (three lines) with:

```
- **Agent explanation evals** are built (`eval_sentences`, `grade_explanation`);
  next is running them over more features and adding a judge for the
  explanations themselves.
```

- [ ] **Step 4: Commit**

```bash
git add scripts/eval_report.py README.md
git commit -m "Add eval_report script; document the explanation eval

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```

---

### Task 7: First real run on 7272, 9435 and 9193

This task is the protocol itself, run by Claude through the MCP server with Andy present. Andy must reconnect the server first (`/mcp`), or the new tools are not visible.

**Files:**
- Modify: `FINDINGS.md` (new section 12)
- Create: `results/explanation_evals.jsonl` (written by the tool; commit it)
- Modify: `README.md` "Findings so far" (one paragraph)

- [ ] **Step 1: Predictions on record**

Before running, write in the conversation what each feature should show, from sections 1, 7 and 8 of FINDINGS: 7272's label ("references to various types of bridges") should already have good recall on held-out positives and good precision on random text; the investigated explanation should mainly improve `peak_hit` and the near-miss rows (overpass, ferry). 9435's label ("San Francisco and its surrounding areas") should fail on agent near-misses containing "Sanitize"/"Sanofi" and on any Bay Area sentence without "San"; the investigated explanation ("the token San, any word") should fix both. 9193's label should do fine; investigating should add little.

- [ ] **Step 2: For each feature, run the protocol**

For each of 7272, 9435, 9193 in turn, through the MCP tools (not `scripts/tool.py`, so the contamination record is live):

1. `eval_sentences(feature)` with defaults.
2. Write the `label` predictions in the conversation from the label only.
3. Investigate: `feature_info`, `run_prompt` on fresh sentences, one `steer` or `ablate` as the causal check.
4. Write the explanation and the `investigated` predictions, plus up to 4 agent sentences.
5. `grade_explanation(feature, explanation, {"label": [...], "investigated": [...]})`.
6. Read the rows; note surprises.

- [ ] **Step 3: Report and write up**

Run: `uv run python scripts/eval_report.py`

Add `## 12. Agent explanation eval: first run` to `FINDINGS.md` with the table from the report (feature, set, n, f1, precision, recall, spearman, peak_hit), the three explanations as written, which predictions from step 1 held, the causal check per feature, and the failures the rows exposed. Add a one-paragraph highlight to README "Findings so far".

- [ ] **Step 4: Commit**

```bash
git add FINDINGS.md README.md results/explanation_evals.jsonl
git commit -m "FINDINGS 12: first explanation-eval run on 7272, 9435, 9193

Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>"
git push
```
