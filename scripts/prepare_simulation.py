"""Build the pools for a simulate-from-explanations batch and write its inputs.

For every feature with stored explanations (a results/*_run_agents.json from
an earlier run), builds the pool, and writes:
- the batch (sentences as agents will see them, plus each sentence's source,
  which agents never see) to --out;
- a copy of scripts/simulate_workflow.js with the batch embedded (no sources)
  to --workflow, ready to run as a Claude Code workflow.

Features whose pool is refused are listed with the reason and left out.
Makes no predictions and grades nothing.

Usage: uv run python scripts/prepare_simulation.py --run blind-5 --explanations results/blind4_run_agents.json
           --out results/blind5_inputs.json --workflow /path/outside/the/repo/blind5_workflow.js
           [--layer 12] [--seed 1] [--n-heldout 4] [--n-wild 6] [--n-neighbor 5] [--n-random 6]
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from interp_playground.backend import Backend  # noqa: E402
from interp_playground.evals import POOL_VERSION  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--explanations", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workflow", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-heldout", type=int, default=4)
    ap.add_argument("--n-wild", type=int, default=6)
    ap.add_argument("--n-neighbor", type=int, default=5)
    ap.add_argument("--n-random", type=int, default=6)
    args = ap.parse_args()

    backend = Backend()
    features, refused = [], []
    for a in json.loads(args.explanations.read_text()):
        f = a["feature"]
        if not a.get("explanation") or not a.get("examples_explanation"):
            refused.append({"feature": f, "reason": "no stored explanation for both arms"})
            continue
        try:
            pool = backend._pool(f, args.layer, args.seed, args.n_heldout, args.n_random, args.n_wild, args.n_neighbor)
        except ValueError as exc:
            refused.append({"feature": f, "reason": str(exc)})
            print(f"{f:6} SKIP {str(exc)[:110]}", flush=True)
            continue
        counts = {s: sum(p["source"] == s for p in pool) for s in ("heldout", "wild", "neighbor", "random")}
        features.append({
            "feature": f, "label": a["label"], "pool": pool,
            "explanations": {"examples": a["examples_explanation"], "investigated": a["explanation"]},
        })
        print(f"{f:6} ok   {counts}", flush=True)

    batch = {"run": args.run, "layer": args.layer, "seed": args.seed, "pool_version": POOL_VERSION,
             "n_heldout": args.n_heldout, "n_wild": args.n_wild, "n_neighbor": args.n_neighbor,
             "n_random": args.n_random, "explanations_from": str(args.explanations),
             "features": features, "refused": refused}
    args.out.write_text(json.dumps(batch, indent=1, ensure_ascii=False) + "\n")

    data = {"run": args.run, "layer": args.layer, "features": [
        {"feature": f["feature"], "label": f["label"], "explanations": f["explanations"],
         "sentences": [{"id": p["id"], "text": p["text"]} for p in f["pool"]]}
        for f in features]}
    template = (ROOT / "scripts" / "simulate_workflow.js").read_text()
    marker = "const DATA = /*DATA*/ null"
    assert template.count(marker) == 1
    # ASCII only: the workflow runner refuses a script with raw control characters.
    args.workflow.write_text(template.replace(marker, "const DATA = " + json.dumps(data, ensure_ascii=True)))
    print(f"{len(features)} features ready, {len(refused)} refused; batch -> {args.out}; workflow -> {args.workflow}")


if __name__ == "__main__":
    main()
