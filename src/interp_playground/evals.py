"""Pure pieces of the agent explanation eval: sentence pool, matching, scoring.

The backend supplies texts and ground-truth activations; nothing here
touches the model or the network, so it all runs in unit tests.
"""

import math
import random
import statistics
from collections.abc import Callable
from typing import Any

# feature_info shows this many top examples; the rest are held out for the eval.
EXAMPLES_SHOWN = 8
# Actual activation / max_act at or above this counts as firing.
FIRE_REL = 0.1
# A prediction (0-10) at or above this predicts firing. This is the primary
# definition, kept at 2 so every run is comparable.
PREDICT_FIRE = 2
# The prediction that corresponds to FIRE_REL on the 0-10 scale. F1 at this
# threshold is reported beside the primary one ("f1_at_1"): with PREDICT_FIRE
# alone, a correct prediction of a firing at 0.1-0.2 of max counts as a miss.
PREDICT_FIRE_MATCHED = 1
MAX_EXPECTED = 10
MAX_AGENT = 4
MAX_SETS = 4
SOURCES = ("heldout", "wild", "neighbor", "random", "agent")
# A feature is only evaluable if at least this many held-out windows fire.
MIN_FIRING_HELDOUT = 3
# Bumped whenever pool construction changes, and written to every record:
# 1 = next-best examples, fixed window; 2 = spread examples, random peak
# position, random pool keyed on the feature; 3 = held-out windows verified to
# fire; 4 = copies of the examples feature_info shows are never held out;
# 5 = whitespace kept as the model saw it, a window must keep KEEP_RECORDED of
# the activation recorded for it in full context, and a pool can also carry
# "wild" and "neighbor" sentences (see wild.py); 6 = a feature is refused when
# most of the examples feature_info shows peak on <bos>.
POOL_VERSION = 6
# A window is only used if the feature reaches this share of the activation
# recorded for the same tokens in their full context. Below it, what the
# feature responds to was cut away and the window fires, if at all, on noise.
KEEP_RECORDED = 0.5
# Activation / max_act at or above this is a "strong" firing; from FIRE_REL up
# to it is "weak". Used to pick wild sentences and to report recall by strength.
WILD_STRONG_REL = 0.3


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


def pick_firing(start: int, stop: int, n: int, fires: Callable[[int], bool]) -> list[int]:
    """Up to `n` indices in [start, stop) for which `fires` holds, spread over the range.

    Starts from the evenly spread picks; a pick that does not fire is replaced
    by its nearest neighbour that does (i+1, i-1, i+2, ...). `fires` is called
    at most once per index. Returns fewer than `n` when too few fire.
    """
    known: dict[int, bool] = {}

    def check(i: int) -> bool:
        if i not in known:
            known[i] = fires(i)
        return known[i]

    chosen: list[int] = []
    for target in spread_indices(start, stop, n):
        by_distance = [target] + [
            i for d in range(1, stop - start) for i in (target + d, target - d) if start <= i < stop
        ]
        pick = next((i for i in by_distance if i not in chosen and check(i)), None)
        if pick is not None:
            chosen.append(pick)
    return sorted(chosen)


WINDOW_TOKENS = 16
MIN_AFTER = 2


def window_offsets(rng: random.Random, total: int = WINDOW_TOKENS, min_after: int = MIN_AFTER) -> tuple[int, int]:
    """(before, after) token counts around the peak, with the peak's position drawn at random.

    A fixed offset would tell the agent where the trigger is in every sentence.
    """
    after = rng.randint(min_after, total - min_after)
    return total - after, after


def _pool_only(scored: dict[str, Any], sources: tuple[str, ...] | None = None) -> dict[str, Any]:
    """A set's scores over pool sentences alone, dropping the agent's own.

    Agent sentences exist only in the investigated set, so including them
    would compare the two sets on different sentences. With `sources`, only
    pool sentences from those sources are scored.
    """
    rows = scored.get("rows")
    if not rows:
        return scored
    # Always rescored from the rows, so records written before a metric
    # existed are compared on it too.
    if sources is None:
        return score_set([dict(r) for r in rows if r["source"] != "agent"])
    return score_set([dict(r) for r in rows if r["source"] in sources])


def paired_summary(
    records: list[dict[str, Any]], base: str = "label", other: str = "investigated",
    sources: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """`other` minus `base` (default: investigated minus label) across records
    that scored both sets, over the pool sentences both sets predicted (agent
    sentences excluded). `sources` restricts the comparison to pool sentences
    from those sources, e.g. ("wild", "neighbor", "random").

    Per metric: mean delta, wins/ties/losses, and
    a two-sided exact sign test over the non-tied pairs. Wins are `other` over `base`.
    """
    out: dict[str, Any] = {"n": 0}
    pairs = [
        {name: _pool_only(r["sets"][name], sources) for name in (base, other)}
        for r in records if base in r.get("sets", {}) and other in r["sets"]
    ]
    for metric in ("f1", "f1_at_1", "spearman", "peak_hit"):
        deltas = [
            s[other][metric] - s[base][metric]
            for s in pairs
            if s[base].get(metric) is not None and s[other].get(metric) is not None
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
        if metric != "peak_hit":  # undefined wherever no peak was named: not the comparison's n
            out["n"] = max(out["n"], len(deltas))
    return out


def normalize(text: str) -> str:
    """Whitespace-insensitive form used for every text comparison in the eval."""
    return " ".join(text.split())


def duplicates_shown(window: str, shown_texts: list[str]) -> bool:
    """Whether a held-out window is part of an example the agent can see in full.

    Neuronpedia's example lists repeat texts, so "beyond the first 8" does not
    by itself mean unseen.
    """
    w = normalize(window)
    return bool(w) and any(w in normalize(t) for t in shown_texts)


def relative_activation(act: float, max_act: float) -> float:
    """act / max_act with the same guard summarize uses, so dead features don't divide by zero."""
    return act / max(max_act, 1e-6)


def window_usable(act: float, rel: float, recorded: float | None) -> bool:
    """Is a short window a fair stand-in for the context it was cut from?

    It must fire, and keep KEEP_RECORDED of the activation recorded for the
    same tokens in their full context (if one was recorded).
    """
    return rel >= FIRE_REL and (recorded is None or act >= KEEP_RECORDED * recorded)


def take_strata(strong: list[Any], weak: list[Any], n: int, ok: Callable[[Any], bool]) -> list[Any]:
    """Up to `n` candidates that pass `ok`: half from `strong` (rounded down),
    the rest from `weak`, each in the order given. A stratum that runs short
    is made up from the other. `ok` is called only on candidates that are needed.
    """
    stop = object()

    def take(it: Any, k: int) -> list[Any]:
        got: list[Any] = []
        while len(got) < k:
            c = next(it, stop)
            if c is stop:
                break
            if ok(c):
                got.append(c)
        return got

    s, w = iter(strong), iter(weak)
    out = take(s, n // 2)
    out += take(w, n - len(out))
    return out + take(s, n - len(out))


def build_pool(
    heldout: list[str], random_texts: list[str], seed: int,
    wild: list[str] | tuple[str, ...] = (), neighbor: list[str] | tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Shuffle the sources together; ids are 0..n-1 in shuffled order.

    Duplicate texts are dropped, keeping the first occurrence (held-out, then
    wild, neighbor, random), so no two ids ever carry the same sentence.
    """
    items, seen = [], set()
    tagged = [(heldout, "heldout"), (wild, "wild"), (neighbor, "neighbor"), (random_texts, "random")]
    for text, source in [(t, source) for texts, source in tagged for t in texts]:
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
        # Agent sentences are run exactly as written: features can depend on
        # newlines and indentation. Only comparisons are whitespace-insensitive.
        pid, raw = pred.get("id"), pred.get("text") or ""
        text = normalize(raw)
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
            base = {"id": None, "text": raw, "source": "agent"}
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
            # A peak can be a whitespace token ("\n", an indent): keep those as given.
            "on": (on or "").strip() or on or None,
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
    spacing and inflection: "bridge" matches " bridges" and vice versa. A
    whitespace token only matches whitespace of the same kind ("\n" matches
    "\n\n", not an indent)."""
    if not on:
        return False
    a = on.strip().lstrip("▁").lower()
    b = peak_token.strip().lstrip("▁").lower()
    if not a and not b:
        return on.startswith(peak_token) or peak_token.startswith(on)
    return bool(a) and bool(b) and (a.startswith(b) or b.startswith(a))


def spearman(expected: list[float], rel: list[float]) -> float | None:
    """Rank correlation, with sub-threshold activations clamped to 0 first so
    ordering among non-firing sentences doesn't count. None if undefined."""
    ys = [0.0 if r < FIRE_REL else r for r in rel]
    try:
        return round(statistics.correlation(list(map(float, expected)), ys, method="ranked"), 3)
    except statistics.StatisticsError:
        return None


def _prf(rows: list[dict[str, Any]], threshold: float = PREDICT_FIRE) -> dict[str, float | None]:
    """Precision, recall and F1 of "expected >= threshold" against "fired"."""
    tp = sum(r["fired"] and r["expected"] >= threshold for r in rows)
    fp = sum((not r["fired"]) and r["expected"] >= threshold for r in rows)
    fn = sum(r["fired"] and r["expected"] < threshold for r in rows)
    p = tp / (tp + fp) if tp + fp else None
    rc = tp / (tp + fn) if tp + fn else None
    # Undefined only when nothing fired. Predicting nothing where the feature
    # does fire scores 0: leaving it undefined would drop the clearest failures
    # from every comparison.
    f1 = None if rc is None else 2 * tp / (2 * tp + fp + fn)
    rnd = lambda v: None if v is None else round(v, 3)
    return {"precision": rnd(p), "recall": rnd(rc), "f1": rnd(f1)}


def score_set(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Score one prediction set. Rows need expected, rel, peak_token, on, source;
    fired / predicted_fired / peak_hit are added to each row in place.

    `f1` (and precision, recall, the rows' predicted_fired) use the primary
    threshold PREDICT_FIRE; `f1_at_1` uses PREDICT_FIRE_MATCHED. `by_strength`
    is recall, at the primary threshold, on weak and on strong firings."""
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
    by_strength = {}
    for name, group in (
        ("weak", [r for r in rows if r["fired"] and r["rel"] < WILD_STRONG_REL]),
        ("strong", [r for r in rows if r["rel"] >= WILD_STRONG_REL]),
    ):
        if group:
            by_strength[name] = {"n": len(group), "recall": _prf(group)["recall"]}
    return {
        "n": len(rows),
        "spearman": spearman([r["expected"] for r in rows], [r["rel"] for r in rows]),
        **_prf(rows),
        "f1_at_1": _prf(rows, PREDICT_FIRE_MATCHED)["f1"],
        "by_strength": by_strength,
        "peak_hit": round(sum(hits) / len(hits), 3) if hits else None,
        "by_source": by_source,
        "rows": rows,
    }
