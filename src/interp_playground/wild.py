"""Sentences from text that was not picked for a feature, for the explanation eval.

A feature's Neuronpedia examples are its strongest activations, and the eight
an agent is shown look like the ones held out. `scripts/scan_wild.py` looks
for two other kinds of sentence and writes them here, one file per feature:

- "strong" and "weak": windows of ordinary text where the feature fires, in a
  seeded random order (not sorted by strength). Each carries `act`, the
  activation in the full context the window was cut from.
- "neighbor": windows where one of the feature's nearest features (by decoder
  cosine) fires. Whether the feature itself fires there is not looked at.

Nothing here touches the model.
"""

import json
import os
from pathlib import Path
from typing import Any

# Override with INTERP_PLAYGROUND_WILD (tests and the smoke test do).
WILD_DIR = Path(
    os.environ.get("INTERP_PLAYGROUND_WILD", Path(__file__).resolve().parents[2] / "results" / "wild")
)


def wild_path(layer: int, feature: int) -> Path:
    return WILD_DIR / str(layer) / f"{feature}.json"


def load_wild(layer: int, feature: int) -> dict[str, Any] | None:
    """The scan's candidates for a feature, or None if it has not been scanned."""
    path = wild_path(layer, feature)
    return json.loads(path.read_text()) if path.exists() else None


def save_wild(layer: int, feature: int, found: dict[str, Any]) -> Path:
    path = wild_path(layer, feature)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(found, indent=1, ensure_ascii=False) + "\n")
    return path
