"""Owns the model, SAEs and feature metadata, and runs experiments on them.

One instance per server process: the model stays loaded between calls.
Everything that touches the model goes through `self._lock`, because MCP
runs sync tools on worker threads.
"""

import threading
from typing import Any

import nnsight
import torch
from nnsight import LanguageModel
from sae_lens import SAE

from interp_playground import neuronpedia
from interp_playground.loading import load_model, load_sae

NUM_LAYERS = 26
# Features active on more than this fraction of tokens carry little meaning
# (e.g. 2620 fires on 98% of tokens) and crowd out everything else.
DENSE_THRESHOLD = 0.1


class Backend:
    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._lock = threading.Lock()
        self._model: LanguageModel | None = None
        self._saes: dict[int, SAE] = {}
        self._tables: dict[int, dict[int, neuronpedia.FeatureMeta]] = {}
        # Per layer: (max activation per feature, is-dense mask), as tensors.
        self._stats: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def warm_up(self, layers: list[int]) -> None:
        with self._lock:
            self._get_model()
            for layer in layers:
                self._get_sae(layer)
                self._get_table(layer)

    def run_prompt(
        self, prompt: str, layers: list[int], top_k: int, max_new_tokens: int, include_dense: bool
    ) -> dict[str, Any]:
        for layer in layers:
            if not 0 <= layer < NUM_LAYERS:
                raise ValueError(f"layer {layer} out of range 0-{NUM_LAYERS - 1}")

        with self._lock:
            model = self._get_model()
            ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
            prompt_len = ids.shape[1]
            if max_new_tokens > 0:
                with torch.no_grad(), model.generate(
                    ids, max_new_tokens=max_new_tokens, do_sample=False
                ) as tracer:
                    ids = tracer.result.save()
                ids = ids.cpu()

            with torch.no_grad(), model.trace(ids):
                resid = nnsight.save([model.model.layers[layer].output for layer in layers])

            per_layer = {}
            for layer, r in zip(layers, resid):
                with torch.no_grad():
                    acts = self._get_sae(layer).encode(r[0]).cpu()
                per_layer[str(layer)] = self._summarize(
                    acts, ids[0].tolist(), layer, top_k, include_dense
                )

        tok = model.tokenizer
        return {
            "prompt": prompt,
            "completion": tok.decode(ids[0, prompt_len:]) if max_new_tokens > 0 else None,
            "layers": per_layer,
        }

    def _summarize(
        self,
        acts: torch.Tensor,
        ids: list[int],
        layer: int,
        top_k: int,
        include_dense: bool,
    ) -> dict[str, Any]:
        # Position 0 is <bos>: huge norm, not in Gemma Scope's training data.
        acts = acts[1:]
        tokens = [self._model.tokenizer.decode(i) for i in ids[1:]]
        table = self._get_table(layer)

        max_act, dense = self._stats[layer]
        if not include_dense:
            acts = acts.masked_fill(dense, 0.0)
        # Rank by activation relative to the feature's own max, so a strong
        # firing of a quiet feature beats a middling firing of a loud one.
        rel = acts / max_act.clamp(min=1e-6)

        def entry(feature: int, pos: int) -> dict[str, Any]:
            return {
                "feature": feature,
                "act": round(acts[pos, feature].item(), 2),
                "rel": round(rel[pos, feature].item(), 2),
            }

        per_token = []
        for pos, token in enumerate(tokens):
            top = rel[pos].topk(top_k)
            feats = [entry(f, pos) for v, f in zip(top.values.tolist(), top.indices.tolist()) if v > 0]
            per_token.append({"pos": pos + 1, "token": token, "features": feats})

        # Whole-text view: each feature at its strongest position.
        best_rel, best_pos = rel.max(dim=0)
        summary = []
        for feature in best_rel.topk(min(10, int((best_rel > 0).sum()))).indices.tolist():
            pos = best_pos[feature].item()
            meta = table[feature]
            summary.append(
                {
                    **entry(feature, pos),
                    "at_pos": pos + 1,
                    "at_token": tokens[pos],
                    "label": meta.label,
                    "density": round(meta.density, 5),
                    "promotes": meta.promotes,
                }
            )

        return {
            "top_features": summary,
            "tokens": per_token,
            "dense_features_hidden": 0 if include_dense else int(dense.sum()),
        }

    def _get_model(self) -> LanguageModel:
        if self._model is None:
            self._model = load_model(self._device)
        return self._model

    def _get_sae(self, layer: int) -> SAE:
        if layer not in self._saes:
            self._saes[layer] = load_sae(layer, self._device)
        return self._saes[layer]

    def _get_table(self, layer: int) -> dict[int, neuronpedia.FeatureMeta]:
        if layer not in self._tables:
            table = neuronpedia.load_table(layer)
            metas = [table[i] for i in range(len(table))]
            self._tables[layer] = table
            self._stats[layer] = (
                torch.tensor([m.max_act for m in metas]),
                torch.tensor([m.density for m in metas]) > DENSE_THRESHOLD,
            )
        return self._tables[layer]
