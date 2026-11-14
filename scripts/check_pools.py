"""Build the eval pool for every selected feature and report which are evaluable.

Run before a batch so refusals are known and logged up front, not discovered
mid-run. Builds pools only; makes no predictions and grades nothing.

Usage: uv run python scripts/check_pools.py [--seed 1] [--n-heldout 6] [--n-random 10]
Reads results/blind_features.json; writes results/blind_pools.json.
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
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-heldout", type=int, default=6)
    ap.add_argument("--n-random", type=int, default=10)
    args = ap.parse_args()

    selection = json.loads((ROOT / "results" / "blind_features.json").read_text())
    backend = Backend()
    out = []
    for f in selection["features"]:
        row = {"feature": f["feature"], "label": f["label"]}
        try:
            pool = backend._pool(f["feature"], selection["layer"], args.seed, args.n_heldout, args.n_random)
            row |= {"evaluable": True, "n": len(pool),
                    "n_heldout": sum(p["source"] == "heldout" for p in pool)}
        except ValueError as exc:
            row |= {"evaluable": False, "reason": str(exc)}
        out.append(row)
        print(f"{row['feature']:6} {'ok  ' if row['evaluable'] else 'SKIP'} "
              f"{row.get('n_heldout', '-')} held-out  {row.get('reason', '')[:90]}", flush=True)
    result = {"pool_version": POOL_VERSION, "layer": selection["layer"], "seed": args.seed,
              "n_heldout": args.n_heldout, "n_random": args.n_random, "features": out}
    (ROOT / "results" / "blind_pools.json").write_text(json.dumps(result, indent=1) + "\n")
    ok = [r for r in out if r["evaluable"]]
    print(f"{len(ok)} of {len(out)} evaluable; "
          f"{sum(r['n_heldout'] == args.n_heldout for r in ok)} with a full held-out set")


if __name__ == "__main__":
    main()
