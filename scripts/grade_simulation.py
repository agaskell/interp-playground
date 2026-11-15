"""Grade a simulate-from-explanations batch: the predictions a workflow run of
scripts/simulate_workflow.js returned, against the pools prepare_simulation.py built.

Calls the same grading code as the MCP tool, once per feature, and appends
one record per feature to results/explanation_evals.jsonl. A feature is
graded only if all three arms returned a complete prediction set.

Usage: uv run python scripts/grade_simulation.py --inputs results/blind5_inputs.json --predictions PATH
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from interp_playground.backend import Backend  # noqa: E402
from interp_playground.evals import POOL_VERSION  # noqa: E402

ARMS = ("label", "examples", "investigated")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True, help="JSON list of {feature, arm, predictions}")
    args = ap.parse_args()

    batch = json.loads(args.inputs.read_text())
    if batch["pool_version"] != POOL_VERSION:
        raise SystemExit(f"inputs were built with pool version {batch['pool_version']}, the code is at {POOL_VERSION}")
    sets: dict[int, dict[str, list]] = {}
    for r in json.loads(args.predictions.read_text()):
        if not r.get("failed"):
            sets.setdefault(r["feature"], {})[r["arm"]] = r["predictions"]

    backend = Backend()
    for f in batch["features"]:
        got = sets.get(f["feature"], {})
        if set(got) != set(ARMS):
            print(f"{f['feature']:6} NOT GRADED: missing {sorted(set(ARMS) - set(got))}", flush=True)
            continue
        text = {p["id"]: p["text"] for p in f["pool"]}
        predictions = {
            arm: [{"id": p["id"], "text": text[p["id"]], "expected": p["expected"], **({"on": p["on"]} if p.get("on") else {})}
                  for p in got[arm]]
            for arm in ARMS
        }
        rec = backend.grade_explanation(
            f["feature"], batch["layer"], f["explanations"]["investigated"], predictions,
            batch["seed"], batch["n_heldout"], batch["n_random"], run=batch["run"],
            n_wild=batch["n_wild"], n_neighbor=batch["n_neighbor"],
        )
        print(f"{f['feature']:6} " + "  ".join(f"{arm} F1 {rec['sets'][arm]['f1']}" for arm in ARMS), flush=True)


if __name__ == "__main__":
    main()
