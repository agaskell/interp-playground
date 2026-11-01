"""Summarise explanation-eval results: one line per prediction set, with the
label-vs-investigated F1 difference per run.

Usage: uv run python scripts/eval_report.py [results/explanation_evals.jsonl]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from interp_playground.evals import paired_summary  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def fmt(x: float | None) -> str:
    return "    -" if x is None else f"{x:5.2f}"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "results" / "explanation_evals.jsonl"
    if not path.exists():
        sys.exit(f"no results at {path}")
    print(f"{'time':19} {'feat':>5} {'set':13} {'n':>3} {'f1':>5} {'prec':>5} {'rec':>5} "
          f"{'rho':>5} {'peak':>5} {'cont':>4}  label")
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for rec in records:
        for name, s in rec["sets"].items():
            print(f"{rec['time'][:19]:19} {rec['feature']:>5} {name[:13]:13} {s['n']:>3} "
                  f"{fmt(s['f1'])} {fmt(s['precision'])} {fmt(s['recall'])} "
                  f"{fmt(s['spearman'])} {fmt(s['peak_hit'])} {len(rec['contaminated']):>4}  "
                  f"{(rec['label'] or '')[:40]}")
        sets = rec["sets"]
        if "label" in sets and "investigated" in sets:
            d = {}
            for m in ("f1", "spearman"):
                a, b = sets["label"][m], sets["investigated"][m]
                d[m] = None if a is None or b is None else b - a
            print(f"{'':19} {'':>5} {'delta':13} {'':>3} {fmt(d['f1'])} {'':>5} {'':>5} {fmt(d['spearman'])}")
    s = paired_summary(records)
    if s["n"]:
        print(f"\npaired, investigated minus label, n={s['n']}:")
        for m in ("f1", "spearman"):
            r = s[m]
            print(f"  {m:9} mean {fmt(r['mean_delta'])}  wins/ties/losses {r['wins']}/{r['ties']}/{r['losses']}"
                  f"  sign test p={'-' if r['sign_p'] is None else f'{r['sign_p']:.3f}'}")


if __name__ == "__main__":
    main()
