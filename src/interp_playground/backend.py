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
from interp_playground.summarize import find_position_artifacts, summarize

NUM_LAYERS = 26
# Features active on more than this fraction of tokens carry little meaning
# (e.g. 2620 fires on 98% of tokens) and crowd out everything else.
DENSE_THRESHOLD = 0.1
# Varied first tokens (prose, code, digits, markup, Spanish) for detecting
# features that respond to "first position" rather than content.
ARTIFACT_PROBES = [
    "The weather was cold.",
    "def main():\n    pass",
    "1987 was a long year.",
    "Bananas are yellow.",
    "He said nothing at all.",
    "<!DOCTYPE html>\n<html>",
    "Q: What is two plus two?",
    "¿Dónde está la biblioteca?",
]


class Backend:
    def __init__(self, device: str | None = None) -> None:
        self._device = device
        self._lock = threading.Lock()
        self._model: LanguageModel | None = None
        self._saes: dict[int, SAE] = {}
        self._tables: dict[int, dict[int, neuronpedia.FeatureMeta]] = {}
        # Per layer: max activation per feature, and masks of features hidden
        # from rankings by default ("dense", "position_artifact").
        self._max_act: dict[int, torch.Tensor] = {}
        self._hidden: dict[int, dict[str, torch.Tensor]] = {}

    def warm_up(self, layers: list[int]) -> None:
        with self._lock:
            self._get_model()
            for layer in layers:
                self._prepare_layer(layer)

    def run_prompt(
        self,
        prompt: str,
        layers: list[int],
        top_k: int,
        features: list[int],
        max_new_tokens: int,
        include_hidden: bool,
    ) -> dict[str, Any]:
        for layer in layers:
            if not 0 <= layer < NUM_LAYERS:
                raise ValueError(f"layer {layer} out of range 0-{NUM_LAYERS - 1}")

        with self._lock:
            model = self._get_model()
            for layer in layers:
                self._prepare_layer(layer)
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

            # Position 0 is <bos>: huge norm, not in Gemma Scope's training data.
            tokens = [model.tokenizer.decode(i) for i in ids[0, 1:].tolist()]
            per_layer = {}
            for layer, r in zip(layers, resid):
                with torch.no_grad():
                    acts = self._saes[layer].encode(r[0, 1:]).cpu()
                per_layer[str(layer)] = summarize(
                    acts,
                    tokens,
                    self._max_act[layer],
                    self._hidden[layer],
                    self._tables[layer],
                    top_k=top_k,
                    watch=features,
                    include_hidden=include_hidden,
                )

        tok = model.tokenizer
        return {
            "prompt": prompt,
            "completion": tok.decode(ids[0, prompt_len:]) if max_new_tokens > 0 else None,
            "layers": per_layer,
        }

    def _get_model(self) -> LanguageModel:
        if self._model is None:
            self._model = load_model(self._device)
        return self._model

    def _prepare_layer(self, layer: int) -> None:
        """Load the SAE and feature metadata for a layer, and find its artifacts."""
        if layer in self._hidden:
            return
        sae = load_sae(layer, self._device)
        table = neuronpedia.load_table(layer)
        metas = [table[i] for i in range(len(table))]

        model = self._get_model()
        probe_acts = []
        for probe in ARTIFACT_PROBES:
            with torch.no_grad(), model.trace(probe):
                r = model.model.layers[layer].output.save()
            with torch.no_grad():
                probe_acts.append(sae.encode(r[0, 1:]).cpu())

        self._saes[layer] = sae
        self._tables[layer] = table
        self._max_act[layer] = torch.tensor([m.max_act for m in metas])
        self._hidden[layer] = {
            "dense": torch.tensor([m.density for m in metas]) > DENSE_THRESHOLD,
            "position_artifact": find_position_artifacts(probe_acts),
        }
