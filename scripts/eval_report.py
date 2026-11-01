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
