"""Pick blind features for an explanation-eval run and write the selection.

Blind = never mentioned in FINDINGS.md or README.md, not hidden (dense or
position artifact), and with enough Neuronpedia examples to hold some out.
The selection is a function of the seed, so it can be re-derived.

Usage: uv run python scripts/select_features.py [--n 25] [--seed 1] [--n-heldout 6]
Writes results/blind_features.json.
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from interp_playground import neuronpedia  # noqa: E402
from interp_playground.backend import D_SAE, Backend  # noqa: E402
from interp_playground.evals import EXAMPLES_SHOWN  # noqa: E402

LAYER = 12


def mentioned_features() -> set[int]:
    """Every integer 0-16383 that appears in the write-ups: over-inclusive on purpose."""
    text = (ROOT / "FINDINGS.md").read_text() + (ROOT / "README.md").read_text()
    return {int(m) for m in re.findall(r"\b\d{1,5}\b", text) if int(m) < D_SAE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-heldout", type=int, default=6)
    args = ap.parse_args()

    excluded = mentioned_features()
    backend = Backend()
    backend._prepare_layer(LAYER)
    masks = backend._hidden[LAYER]
    hidden = (masks["dense"] | masks["position_artifact"]).tolist()
    table = neuronpedia.load_table(LAYER)

    rng = random.Random(f"blind:{args.seed}")
    order = list(range(D_SAE))
    rng.shuffle(order)
    chosen, skipped = [], {"excluded": 0, "hidden": 0, "no_label": 0, "few_examples": 0, "unreachable": 0}
    for f in order:
        if len(chosen) >= args.n:
            break
        if f in excluded:
            skipped["excluded"] += 1
            continue
        if hidden[f]:
            skipped["hidden"] += 1
            continue
        meta = table.get(f)
        label = meta.label if meta else None
        if not label:
            skipped["no_label"] += 1
            continue
        examples = neuronpedia.load_examples(LAYER, f)
        if examples is None:
            skipped["unreachable"] += 1
            continue
        if len(examples) < EXAMPLES_SHOWN + args.n_heldout:
            skipped["few_examples"] += 1
            continue
        chosen.append({"feature": f, "label": label, "density": meta.density, "n_examples": len(examples)})
        print(f"{len(chosen):3} {f:6} ({len(examples)} ex) {label}", flush=True)

    out = {
        "layer": LAYER, "seed": args.seed, "n_heldout": args.n_heldout,
        "criteria": "not mentioned in FINDINGS/README, not hidden, labelled, "
                    f">= {EXAMPLES_SHOWN + args.n_heldout} Neuronpedia examples; order = Random('blind:<seed>').shuffle",
        "n_excluded_ids": len(excluded), "skipped": skipped, "features": chosen,
    }
    (ROOT / "results" / "blind_features.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {len(chosen)} features; skipped {skipped}")


if __name__ == "__main__":
    main()
