"""Pure pieces of the agent explanation eval: sentence pool, matching, scoring.

The backend supplies texts and ground-truth activations; nothing here
touches the model or the network, so it all runs in unit tests.
"""

import math
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
    # Keyed on the target feature too, so two features never share a random pool.
    return random.Random(f"{seed}:{exclude}").sample(candidates, min(n, len(candidates)))


def spread_indices(start: int, stop: int, n: int) -> list[int]:
    """`n` distinct indices spread evenly over [start, stop), first and last included."""
    if stop - start <= n:
        return list(range(start, stop))
    if n == 1:
        return [start]
    return sorted({start + round(k * (stop - 1 - start) / (n - 1)) for k in range(n)})


WINDOW_TOKENS = 16
MIN_AFTER = 2


def window_offsets(rng: random.Random, total: int = WINDOW_TOKENS, min_after: int = MIN_AFTER) -> tuple[int, int]:
    """(before, after) token counts around the peak, with the peak's position drawn at random.

    A fixed offset would tell the agent where the trigger is in every sentence.
    """
    after = rng.randint(min_after, total - min_after)
    return total - after, after


def paired_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Label-vs-investigated deltas across records that scored both sets.

    Per metric: mean delta, wins/ties/losses for investigated over label, and
    a two-sided exact sign test over the non-tied pairs.
    """
    out: dict[str, Any] = {"n": 0}
    pairs = [r["sets"] for r in records if "label" in r.get("sets", {}) and "investigated" in r["sets"]]
    for metric in ("f1", "spearman"):
        deltas = [
            s["investigated"][metric] - s["label"][metric]
            for s in pairs
            if s["label"].get(metric) is not None and s["investigated"].get(metric) is not None
        ]
        wins = sum(d > 1e-9 for d in deltas)
        losses = sum(d < -1e-9 for d in deltas)
        ties = len(deltas) - wins - losses
        m = wins + losses
        if m:
            k = min(wins, losses)
            sign_p = min(1.0, 2 * sum(math.comb(m, j) for j in range(k + 1)) / 2**m)
        else:
            sign_p = None
        out[metric] = {
            "n": len(deltas),
            "mean_delta": statistics.fmean(deltas) if deltas else None,
            "wins": wins, "ties": ties, "losses": losses, "sign_p": sign_p,
        }
        out["n"] = max(out["n"], len(deltas))
    return out


def normalize(text: str) -> str:
    """Whitespace-insensitive form used for every text comparison in the eval."""
    return " ".join(text.split())


def relative_activation(act: float, max_act: float) -> float:
    """act / max_act with the same guard summarize uses, so dead features don't divide by zero."""
    return act / max(max_act, 1e-6)


def build_pool(heldout: list[str], random_texts: list[str], seed: int) -> list[dict[str, Any]]:
    """Shuffle the sources together; ids are 0..n-1 in shuffled order.

    Duplicate texts are dropped, keeping the first occurrence (held-out
    before random), so no two ids ever carry the same sentence.
    """
    items, seen = [], set()
    for text, source in [(t, "heldout") for t in heldout] + [(t, "random") for t in random_texts]:
        if normalize(text) not in seen:
            seen.add(normalize(text))
            items.append((text, source))
    random.Random(seed).shuffle(items)
    return [{"id": i, "text": t, "source": s} for i, (t, s) in enumerate(items)]


def match_predictions(pool: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair predictions with pool sentences by id; anything else is an agent sentence.

    Every pool id must be predicted exactly once. `expected` is coerced to a
    float and clamped to 0-10.
    """
    by_id = {p["id"]: p for p in pool}
    by_text = {normalize(p["text"]): p for p in pool}
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    for pred in predictions:
        pid, text = pred.get("id"), normalize(pred.get("text") or "")
        if pid is None and text in by_text:
            pid = by_text[text]["id"]
        if pid is not None:
            if pid not in by_id:
                raise ValueError(f"unknown sentence id {pid}")
            if pid in seen:
                raise ValueError(f"sentence id {pid} predicted twice")
            base = by_id[pid]
            if text and text != normalize(base["text"]):
                raise ValueError(
                    f"id {pid} is {base['text']!r} in this pool, not {text!r}: "
                    "check seed, n_heldout and n_random match eval_sentences"
                )
            seen.add(pid)
        elif text:
            base = {"id": None, "text": text, "source": "agent"}
        else:
            raise ValueError("each prediction needs an id (pool sentence) or text (agent sentence)")
        if "expected" not in pred:
            raise ValueError(f"prediction for {pid if pid is not None else text!r} has no 'expected'")
        try:
            expected = float(pred["expected"])
        except (TypeError, ValueError):
            expected = math.nan
        if not math.isfinite(expected):
            raise ValueError(f"expected must be a number 0-10, got {pred['expected']!r}")
        on = pred.get("on")
        if on is not None and not isinstance(on, str):
            raise ValueError(f"on must be a string, got {on!r}")
        rows.append({
            **base,
            "expected": min(max(expected, 0.0), float(MAX_EXPECTED)),
            "on": (on or "").strip() or None,
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
