"""Summarise explanation-eval results: one line per prediction set, with the
label-vs-investigated F1 difference per run.

Usage: uv run python scripts/eval_report.py [results/explanation_evals.jsonl] [--run LABEL]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from interp_playground.evals import paired_summary, score_set  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fmt(x: float | None) -> str:
    return "    -" if x is None else f"{x:5.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("path", nargs="?", type=Path, default=ROOT / "results" / "explanation_evals.jsonl")
    ap.add_argument("--run", help="only records tagged with this run label")
    args = ap.parse_args()
    path = args.path
    if not path.exists():
        sys.exit(f"no results at {path}")
    print(f"{'time':19} {'feat':>5} {'set':13} {'n':>3} {'f1':>5} {'f1@1':>5} {'prec':>5} {'rec':>5} "
          f"{'weak':>5} {'strng':>5} {'rho':>5} {'peak':>5} {'cont':>4}  label")
    print("  (f1, prec, rec: a prediction of 2 or more counts as firing; f1@1: 1 or more; "
          "weak, strng: recall on firings below / at or above 0.3 of max)")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if args.run is not None:
        records = [r for r in records if r.get("run", "") == args.run]
    for rec in records:
        for name, stored in rec["sets"].items():
            # Rescored from the rows, so older records show the newer columns too.
            s = score_set([dict(r) for r in stored["rows"]])
            strength = s["by_strength"]
            print(f"{rec['time'][:19]:19} {rec['feature']:>5} {name[:13]:13} {s['n']:>3} "
                  f"{fmt(s['f1'])} {fmt(s['f1_at_1'])} {fmt(s['precision'])} {fmt(s['recall'])} "
                  f"{fmt(strength.get('weak', {}).get('recall'))} {fmt(strength.get('strong', {}).get('recall'))} "
                  f"{fmt(s['spearman'])} {fmt(s['peak_hit'])} {len(rec['contaminated']):>4}  "
                  f"{(rec['label'] or '')[:40]}")
        sets = rec["sets"]
        if "label" in sets and "investigated" in sets:
            d = {}
            for m in ("f1", "spearman"):
                a, b = sets["label"][m], sets["investigated"][m]
                d[m] = None if a is None or b is None else b - a
            print(f"{'':19} {'':>5} {'delta':13} {'':>3} {fmt(d['f1'])} {'':>5} {'':>5} {'':>5} {'':>5} {'':>5} {fmt(d['spearman'])}")
    names = {name for rec in records for name in rec["sets"]}
    contrasts = [("label", "investigated")]
    if "examples" in names:
        contrasts += [("label", "examples"), ("examples", "investigated")]
    # Pools with wild or neighbor sentences are also compared on the sentences
    # that are not top examples, and on the top-example part alone.
    subsets: list[tuple[str, tuple[str, ...] | None]] = [("pool sentences only", None)]
    if any(r["source"] in ("wild", "neighbor") for rec in records for s in rec["sets"].values() for r in s["rows"]):
        subsets += [("wild, neighbor and random sentences", ("wild", "neighbor", "random")),
                    ("held-out and random sentences", ("heldout", "random"))]
    for what, sources in subsets:
        for base, other in contrasts:
            s = paired_summary(records, base, other, sources)
            if not s["n"]:
                continue
            print(f"\npaired, {other} minus {base}, {what}, n={s['n']}:")
            for m in ("f1", "f1_at_1", "spearman", "peak_hit"):
                r = s[m]
                p = "-" if r["sign_p"] is None else f"{r['sign_p']:.3f}"
                print(f"  {m:9} mean {fmt(r['mean_delta'])}  wins/ties/losses {r['wins']}/{r['ties']}/{r['losses']}"
                      f"  sign test p={p}")

if __name__ == "__main__":
    main()
