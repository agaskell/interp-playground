"""Load Gemma 2 2B and its Gemma Scope residual-stream SAEs."""

import torch
from nnsight import LanguageModel
from sae_lens import SAE

MODEL = "google/gemma-2-2b"
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_model(device: str | None = None) -> LanguageModel:
    # fp32: bf16 drifts 1-4% in the residual stream (scripts/check_mps.py),
    # enough to flip JumpReLU features near their thresholds.
    return LanguageModel(
        MODEL,
        device_map=device or default_device(),
        dtype=torch.float32,
        attn_implementation="eager",
        dispatch=True,
    )


def load_sae(layer: int, device: str | None = None) -> SAE:
    """Canonical 16k-wide residual SAE, trained on the output of `layer`."""
    return SAE.from_pretrained(
        SAE_RELEASE, f"layer_{layer}/width_16k/canonical", device=device or default_device()
    )
