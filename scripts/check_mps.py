"""Check that Gemma 2 2B gives the same internals on MPS as on CPU.

CPU fp32 is the reference. For each other config, compares the residual
stream after every decoder layer and the final (soft-capped) logits.
"""

import gc

import nnsight
import torch
from nnsight import LanguageModel

MODEL = "google/gemma-2-2b"
PROMPTS = [
    "The Eiffel Tower is located in the city of",
    "def fibonacci(n):\n    if n <= 1:\n        return",
]
CONFIGS = [
    ("cpu", torch.float32),
    ("mps", torch.float32),
    ("mps", torch.bfloat16),
]


def capture(device: str, dtype: torch.dtype) -> tuple[list[dict], object]:
    model = LanguageModel(
        MODEL,
        device_map=device,
        dtype=dtype,
        attn_implementation="eager",
        dispatch=True,
    )
    layers = model.model.layers
    results = []
    for prompt in PROMPTS:
        with torch.no_grad(), model.trace(prompt):
            # Only objects that are themselves saved leave the trace, so save
            # the list, not just its elements.
            resid = nnsight.save([layer.output for layer in layers])
            logits = model.output.logits.save()
        results.append(
            {
                "resid": [r[0].float().cpu() for r in resid],
                "logits": logits[0].float().cpu(),
            }
        )
    tokenizer = model.tokenizer
    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    return results, tokenizer


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Worst per-token relative error: ||a - b|| / ||ref|| over the last dim."""
    return ((a - b).norm(dim=-1) / b.norm(dim=-1)).max().item()


def compare(name: str, got: list[dict], ref: list[dict], tokenizer) -> None:
    print(f"\n=== {name} vs cpu/float32 ===")
    for prompt, g, r in zip(PROMPTS, got, ref):
        layer_errs = [rel_err(a, b) for a, b in zip(g["resid"], r["resid"])]
        worst = max(range(len(layer_errs)), key=layer_errs.__getitem__)
        print(f"\nprompt: {prompt!r}")
        print(
            f"  resid rel err: layer0 {layer_errs[0]:.2e}, "
            f"layer12 {layer_errs[12]:.2e}, final {layer_errs[-1]:.2e}, "
            f"worst {layer_errs[worst]:.2e} (layer {worst})"
        )

        gl, rl = g["logits"][-1], r["logits"][-1]
        kl = torch.nn.functional.kl_div(
            gl.log_softmax(-1), rl.log_softmax(-1), log_target=True, reduction="sum"
        ).item()
        g_top = gl.topk(5).indices.tolist()
        r_top = rl.topk(5).indices.tolist()
        print(
            f"  next-token logits: max abs diff {(gl - rl).abs().max().item():.3e}, "
            f"KL(ref||got) {kl:.2e}, top-5 match {g_top == r_top}"
        )
        print(f"  top-5: {[tokenizer.decode(t) for t in g_top]}")


def main() -> None:
    captured = {}
    for device, dtype in CONFIGS:
        print(f"running {device}/{str(dtype).removeprefix('torch.')} ...", flush=True)
        captured[(device, dtype)], tokenizer = capture(device, dtype)

    ref = captured[CONFIGS[0]]
    for device, dtype in CONFIGS[1:]:
        name = f"{device}/{str(dtype).removeprefix('torch.')}"
        compare(name, captured[(device, dtype)], ref, tokenizer)


if __name__ == "__main__":
    main()
