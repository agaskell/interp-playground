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
