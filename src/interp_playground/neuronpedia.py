"""Per-feature metadata from Neuronpedia's bulk exports, cached locally.

Labels are auto-generated (gpt-4o-mini reading top activations), so treat them
as hypotheses. Density and max activation come from Neuronpedia's own runs.
"""

import gzip
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

BUCKET = "https://neuronpedia-datasets.s3.us-east-1.amazonaws.com"
MODEL_ID = "gemma-2-2b"
CACHE_DIR = Path.home() / ".cache" / "interp-playground" / "neuronpedia"


@dataclass(frozen=True)
class FeatureMeta:
    label: str | None
    density: float  # fraction of tokens the feature is active on
    max_act: float  # approximate max activation seen by Neuronpedia
    promotes: list[str]  # output tokens the feature's direction boosts most


def sae_id(layer: int) -> str:
    return f"{layer}-gemmascope-res-16k"


def load_table(layer: int) -> dict[int, FeatureMeta]:
    path = CACHE_DIR / f"{sae_id(layer)}.json"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(_download(layer)))
        tmp.rename(path)
    raw = json.loads(path.read_text())
    return {int(i): FeatureMeta(**m) for i, m in raw.items()}


def _download(layer: int) -> dict[str, dict]:
    prefix = f"v1/{MODEL_ID}/{sae_id(layer)}"
    table: dict[str, dict] = {}
    for rec in _records(f"{prefix}/features/"):
        table[rec["index"]] = {
            "label": None,
            "density": rec["frac_nonzero"],
            "max_act": rec["maxActApprox"],
            "promotes": [t.replace("▁", " ") for t in rec["pos_str"][:5]],
        }
    for rec in _records(f"{prefix}/explanations/"):
        if rec["index"] in table and table[rec["index"]]["label"] is None:
            table[rec["index"]]["label"] = rec["description"].strip()
    return table


def _records(prefix: str):
    with urllib.request.urlopen(f"{BUCKET}/?list-type=2&prefix={prefix}") as resp:
        keys = re.findall(r"<Key>([^<]+\.jsonl\.gz)</Key>", resp.read().decode())
    for key in keys:
        with urllib.request.urlopen(f"{BUCKET}/{key}") as resp:
            for line in gzip.decompress(resp.read()).splitlines():
                yield json.loads(line)


def feature_url(layer: int, feature: int) -> str:
    return f"https://www.neuronpedia.org/{MODEL_ID}/{sae_id(layer)}/{feature}"


def fetch_examples(layer: int, feature: int, n: int = 8, context: int = 12) -> list[dict] | None:
    """Top activating examples from Neuronpedia's API, cached on disk.

    Each example: the max activation, and a snippet with the peak token in
    [[brackets]]. Returns None if Neuronpedia is unreachable.
    """
    path = CACHE_DIR / "features" / sae_id(layer) / f"{feature}.json"
    if not path.exists():
        url = f"https://www.neuronpedia.org/api/feature/{MODEL_ID}/{sae_id(layer)}/{feature}"
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                raw = json.loads(resp.read())
        except (OSError, ValueError):
            return None
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([
            {"max": a["maxValue"], "tokens": a["tokens"], "peak": a["maxValueTokenIndex"]}
            for a in raw.get("activations", [])
        ]))
    examples = []
    for a in json.loads(path.read_text())[:n]:
        toks, i = a["tokens"], a["peak"]
        snippet = "".join(toks[max(0, i - context):i]) + f"[[{toks[i]}]]" + "".join(toks[i + 1:i + 4])
        examples.append({"max": round(a["max"], 1), "text": snippet.replace("\n", "\\n")})
    return examples
