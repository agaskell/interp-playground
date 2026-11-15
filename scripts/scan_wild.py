"""Find sentences a feature was not picked for: ordinary text where it fires,
and text where its nearest features fire. Writes results/wild/<layer>/<feature>.json
for the eval pool's "wild" and "neighbor" sources (see src/interp_playground/wild.py).

The text scanned is every Neuronpedia example context already in the local
cache: 127-token passages that are top activations of *other* features, so
for the feature being scanned they are text nobody picked. The feature's own
example contexts are left out, and so are its nearest features' (those are
where it is most likely to fire, and they feed the "neighbor" source instead).

For each feature:
- every context is run once through the model; the feature's peak activation
  and its position are recorded;
- contexts where it fires (peak >= FIRE_REL of max) are split into "strong"
  (>= WILD_STRONG_REL) and "weak", shuffled with the seed, and the first
  --keep of each are written as 16-token windows around the peak;
- for each of its 5 nearest features by decoder cosine (the ones feature_info
  lists), windows are cut around that feature's own top examples and kept if
  the *neighbor* still fires in them. What the scanned feature does there is
  not looked at.

Makes no predictions and grades nothing.

Usage: uv run python scripts/scan_wild.py --features results/blind_pools_seed2.json
           [--seed 1] [--keep 20] [--batch 32] [--limit N]
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from interp_playground import neuronpedia  # noqa: E402
from interp_playground.backend import Backend  # noqa: E402
from interp_playground.evals import (  # noqa: E402
    FIRE_REL,
    WILD_STRONG_REL,
    duplicates_shown,
    normalize,
    spread_indices,
    window_offsets,
)
from interp_playground.wild import save_wild  # noqa: E402

NEIGHBORS = 5
PER_NEIGHBOR = 4


def full_text(example: dict) -> str:
    n = len(example["tokens"])
    return neuronpedia.example_text(example, n, n)


def cached_contexts(layer: int) -> list[str]:
    """Every distinct example context in the Neuronpedia cache, in a stable order."""
    folder = neuronpedia.CACHE_DIR / "features" / neuronpedia.sae_id(layer)
    seen, out = set(), []
    for path in sorted(folder.glob("*.json"), key=lambda p: int(p.stem)):
        for example in json.loads(path.read_text()):
            text = full_text(example)
            key = normalize(text)
            if key and key not in seen:
                seen.add(key)
                out.append(text)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True, help="a check_pools.py output; its in-batch features are scanned")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep", type=int, default=20, help="candidates written per stratum")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, help="scan only the first N contexts (for a quick check)")
    args = ap.parse_args()

    selection = json.loads(args.features.read_text())
    layer = selection["layer"]
    targets = [f["feature"] for f in selection["features"] if f.get("in_batch", f.get("evaluable"))]

    backend = Backend()
    model = backend._get_model()
    backend._prepare_layer(layer)
    sae, max_act, tok = backend._saes[layer], backend._max_act[layer], model.tokenizer

    texts = cached_contexts(layer)[: args.limit]
    W = sae.W_dec.detach()
    neighbors = {}
    for f in targets:
        cos = torch.nn.functional.cosine_similarity(W, W[f][None], dim=-1)
        cos[f] = -1
        top = cos.topk(NEIGHBORS)
        neighbors[f] = [(int(i), round(float(v), 3)) for v, i in zip(top.values, top.indices)]

    ids = [tok(t, return_tensors="pt")["input_ids"][0] for t in texts]
    by_len: dict[int, list[int]] = {}
    for i, row in enumerate(ids):
        if len(row) > 2:
            by_len.setdefault(len(row), []).append(i)
    print(f"{len(texts)} contexts, {sum(len(r) for r in ids)} tokens, {len(targets)} features", flush=True)

    def acts_for(batch: torch.Tensor) -> torch.Tensor:
        """Activations of the target features at every position after <bos>: [B, T-1, n_targets]."""
        with torch.no_grad(), model.trace(batch):
            resid = model.model.layers[layer].output.save()
        with torch.no_grad():
            return sae.encode(resid[:, 1:])[..., targets].float().cpu()

    peak_act = torch.zeros(len(texts), len(targets))
    peak_pos = torch.zeros(len(texts), len(targets), dtype=torch.long)
    done, start, checked = 0, time.time(), False
    for length in sorted(by_len):
        rows = by_len[length]
        for k in range(0, len(rows), args.batch):
            chunk = rows[k:k + args.batch]
            acts = acts_for(torch.stack([ids[i] for i in chunk]))
            if not checked and len(chunk) > 1:
                # A batched pass must give what a single pass gives, or nothing below means anything.
                alone = acts_for(ids[chunk[-1]][None])
                worst = float((acts[-1] - alone[0]).abs().max())
                scale = float(alone.abs().max()) or 1.0
                if worst > 0.02 * scale + 0.05:
                    raise SystemExit(f"batched and single activations differ by {worst:.3f} (scale {scale:.1f})")
                print(f"batch check: max difference {worst:.4f} on a scale of {scale:.1f}", flush=True)
                checked = True
            vals, pos = acts.max(dim=1)
            peak_act[chunk], peak_pos[chunk] = vals, pos
            done += len(chunk)
            if done % (args.batch * 25) < len(chunk):
                rate = done / (time.time() - start)
                print(f"  {done}/{len(texts)} contexts, {rate:.0f}/s, about {(len(texts) - done) / rate / 60:.1f} min left", flush=True)

    for col, f in enumerate(targets):
        own = neuronpedia.load_examples(layer, f) or []
        own_texts = [full_text(e) for e in own]
        # Left out of this feature's scan: its own examples and its neighbors'.
        own_keys = {normalize(t) for t in own_texts}
        neighbor_examples = {nb: neuronpedia.load_examples(layer, nb) for nb, _ in neighbors[f]}
        own_keys |= {normalize(full_text(e)) for ex in neighbor_examples.values() for e in ex or []}
        top = float(max_act[f])
        strata: dict[str, list[dict]] = {"strong": [], "weak": []}
        for i, text in enumerate(texts):
            rel = float(peak_act[i, col]) / top
            if rel < FIRE_REL or normalize(text) in own_keys:
                continue
            p = 1 + int(peak_pos[i, col])
            before, after = window_offsets(random.Random(f"{args.seed}:{f}:wild:{i}"))
            window = tok.decode(ids[i][max(1, p - before):p + 1 + after])
            strata["strong" if rel >= WILD_STRONG_REL else "weak"].append({
                "text": window, "act": round(float(peak_act[i, col]), 2), "rel": round(rel, 3),
                "peak_token": tok.decode(ids[i][p]), "context": i,
            })
        n_firing = {name: len(c) for name, c in strata.items()}
        for name, cands in strata.items():
            random.Random(f"{args.seed}:{f}:wild:{name}").shuffle(cands)
            del cands[args.keep:]

        # Neighbor sentences, one neighbor at a time in turn so a short pool covers several.
        per_neighbor, unreachable = [], []
        for nb, cos in neighbors[f]:
            examples = neighbor_examples[nb]
            if examples is None:
                unreachable.append(nb)
                continue
            found = []
            for j in spread_indices(0, len(examples), PER_NEIGHBOR):
                rng = random.Random(f"{args.seed}:{f}:neighbor:{nb}:{j}")
                text = neuronpedia.example_text(examples[j], *window_offsets(rng))
                if not text.strip() or duplicates_shown(text, own_texts):
                    continue
                if backend._window_fires(nb, layer, text, examples[j]["max"]):
                    found.append({"text": text, "of": nb, "cos": cos, "label": backend._tables[layer][nb].label})
            per_neighbor.append(found)
        neighbor = [c for rank in range(PER_NEIGHBOR) for found in per_neighbor for c in found[rank:rank + 1]]

        path = save_wild(layer, f, {
            "feature": f, "layer": layer, "seed": args.seed, "max_act": round(top, 2),
            "contexts_scanned": len(texts) - sum(normalize(t) in own_keys for t in texts),
            "contexts_firing": n_firing,
            "neighbors": [{"feature": nb, "cos": cos} for nb, cos in neighbors[f]],
            "neighbors_unreachable": unreachable,
            **strata, "neighbor": neighbor,
        })
        print(f"{f:6} fires in {n_firing['strong']:5} strong, {n_firing['weak']:5} weak contexts; "
              f"{len(neighbor):2} neighbor sentences{' (unreachable: ' + str(unreachable) + ')' if unreachable else ''} "
              f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
