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
from interp_playground.lens import lens_rows
from interp_playground.loading import load_model, load_sae
from interp_playground.steering import random_direction, steer_metrics
from interp_playground.summarize import find_position_artifacts, summarize

NUM_LAYERS = 26
D_SAE = 16384
# Steering's p_promoted metric uses this many of the feature's top promoted tokens.
NUM_PROMOTED = 10
# Default steering scales, as multiples of the feature's typical max activation.
DEFAULT_SCALE_MULTIPLES = [1, 2, 4]
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

    def steer(
        self,
        prompt: str,
        feature: int,
        layer: int,
        scales: list[float] | None,
        max_new_tokens: int,
        control: bool,
        seed: int,
    ) -> dict[str, Any]:
        self._check(feature, layer)
        with self._lock:
            model = self._get_model()
            self._prepare_layer(layer)
            meta = self._tables[layer][feature]
            if scales is None:
                scales = [round(m * meta.max_act) for m in DEFAULT_SCALE_MULTIPLES]

            # Decoder rows are unit-norm, so a scale is in activation units.
            direction = self._saes[layer].W_dec[feature].detach()
            ctrl = random_direction(direction.shape[0], seed).to(direction.device)
            promoted = (model.lm_head.weight.detach() @ direction).topk(NUM_PROMOTED).indices.tolist()

            ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
            base_probs, resid_norm = self._next_token_probs(ids, layer, None)

            def run(vec: torch.Tensor | None, probs: torch.Tensor) -> dict[str, Any]:
                completion = self._generate(ids, layer, vec, max_new_tokens)
                return {
                    "completion": model.tokenizer.decode(completion),
                    **steer_metrics(base_probs, probs, promoted, completion),
                }

            baseline = run(None, base_probs)
            runs = []
            for scale in scales:
                entry: dict[str, Any] = {
                    "scale": scale,
                    "frac_of_resid": round(abs(scale) / resid_norm, 2),
                }
                arms = [("steered", direction)] + ([("control", ctrl)] if control else [])
                for name, unit in arms:
                    vec = scale * unit
                    entry[name] = run(vec, self._next_token_probs(ids, layer, vec)[0])
                runs.append(entry)

        return {
            "feature": feature,
            "layer": layer,
            "label": meta.label,
            "max_act": meta.max_act,
            "promoted_tokens": [model.tokenizer.decode(t) for t in promoted],
            "resid_norm": round(resid_norm, 1),
            "baseline": baseline,
            "runs": runs,
        }

    def feature_info(self, feature: int, layer: int) -> dict[str, Any]:
        self._check(feature, layer)
        with self._lock:
            model = self._get_model()
            self._prepare_layer(layer)
            meta = self._tables[layer][feature]
            W_dec = self._saes[layer].W_dec.detach()
            direction = W_dec[feature]
            with torch.no_grad():
                logit_effect = model.lm_head.weight.detach() @ direction
                promoted = logit_effect.topk(NUM_PROMOTED).indices.tolist()
                suppressed = (-logit_effect).topk(NUM_PROMOTED).indices.tolist()
                cos = torch.nn.functional.cosine_similarity(W_dec, direction[None], dim=-1)
                cos[feature] = -1
                sim = cos.topk(5)
            hidden = next(
                (name for name, mask in self._hidden[layer].items() if mask[feature]), None
            )
            table = self._tables[layer]
            similar = [
                {"feature": i, "cos": round(v, 3), "label": table[i].label}
                for v, i in zip(sim.values.tolist(), sim.indices.tolist())
            ]
        decode = model.tokenizer.decode
        return {
            "feature": feature,
            "layer": layer,
            "label": meta.label,
            "density": round(meta.density, 5),
            "max_act": meta.max_act,
            "hidden": hidden,
            "promotes": [decode(t) for t in promoted],
            "suppresses": [decode(t) for t in suppressed],
            "similar_features": similar,
            "top_examples": neuronpedia.fetch_examples(layer, feature),
            "url": neuronpedia.feature_url(layer, feature),
        }

    def logit_lens(self, prompt: str, position: int, top_k: int) -> dict[str, Any]:
        with self._lock:
            model = self._get_model()
            ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
            n = ids.shape[1]
            pos = position if position >= 0 else n + position
            if not 1 <= pos < n:
                raise ValueError(f"position {position} out of range for {n - 1} tokens")
            with torch.no_grad(), model.trace(ids):
                resid = nnsight.save([layer.output[0, pos] for layer in model.model.layers])
            with torch.no_grad():
                h = torch.stack(resid)  # [n_layers, d_model]
                logits = model.lm_head(model.model.norm(h))
                cap = model.config.final_logit_softcapping
                logits = torch.tanh(logits / cap) * cap
                probs = logits.float().softmax(-1).cpu()
        tokens = [model.tokenizer.decode(i) for i in ids[0, 1:].tolist()]
        return {
            "tokens": tokens,
            "position": pos - 1,
            "token": tokens[pos - 1],
            "final_prediction": model.tokenizer.decode(int(probs[-1].argmax())),
            "layers": lens_rows(probs, list(range(NUM_LAYERS)), model.tokenizer.decode, top_k),
        }

    def ablate(
        self,
        prompt: str,
        feature: int,
        layer: int,
        value: float | None,
        max_new_tokens: int,
        top_k: int,
    ) -> dict[str, Any]:
        """Remove a feature's contribution (or clamp it to `value`) and measure the effect.

        The intervention is resid - (act - target) * W_dec[feature], so the
        rest of the residual stream is untouched: no SAE reconstruction error
        is introduced.
        """
        self._check(feature, layer)
        with self._lock:
            model = self._get_model()
            self._prepare_layer(layer)
            sae = self._saes[layer]
            direction = sae.W_dec[feature].detach()
            ids = model.tokenizer(prompt, return_tensors="pt")["input_ids"]
            tokens = [model.tokenizer.decode(i) for i in ids[0, 1:].tolist()]

            def patch(resid: torch.Tensor) -> torch.Tensor:
                acts = sae.encode(resid)[..., feature : feature + 1]
                target = 0.0 if value is None else value
                return resid - (acts - target) * direction

            def forward(ablated: bool) -> tuple[torch.Tensor, torch.Tensor]:
                with torch.no_grad(), model.trace(ids):
                    target = model.model.layers[layer]
                    acts = sae.encode(target.output[0])[:, feature].save()
                    if ablated:
                        target.output = patch(target.output)
                    logits = model.output.logits.save()
                return acts.cpu(), logits[0].float().log_softmax(-1).cpu()

            acts, base_lp = forward(False)
            _, abl_lp = forward(True)

            # Log-prob of each actual next token, before and after.
            next_ids = ids[0, 1:]
            base_next = base_lp[:-1].gather(1, next_ids[:, None]).squeeze(1)
            abl_next = abl_lp[:-1].gather(1, next_ids[:, None]).squeeze(1)
            delta = abl_next - base_next
            n_pos = ids.shape[1]
            active = [
                {
                    "pos": p - 1,
                    "token": tokens[p - 1],
                    "act": round(acts[p].item(), 2),
                    # How much the ablation changes log p(next token) here.
                    "next_token_logprob_delta": round(delta[p].item(), 3) if p < n_pos - 1 else None,
                }
                for p in range(1, n_pos)
                if acts[p] > 0
            ]

            def top(lp: torch.Tensor) -> list[list]:
                t = lp.exp().topk(top_k)
                return [[model.tokenizer.decode(i), round(v, 4)] for v, i in zip(t.values.tolist(), t.indices.tolist())]

            kl = (base_lp[-1].exp() * (base_lp[-1] - abl_lp[-1])).sum().item()
            out: dict[str, Any] = {
                "feature": feature,
                "layer": layer,
                "mode": "zero" if value is None else f"clamp to {value}",
                "tokens": tokens,
                "active_positions": active,
                "next_token": {
                    "baseline": top(base_lp[-1]),
                    "ablated": top(abl_lp[-1]),
                    "kl": round(max(kl, 0.0), 4),
                },
            }
            if max_new_tokens > 0:
                out["completion"] = {
                    "baseline": model.tokenizer.decode(self._generate(ids, layer, None, max_new_tokens)),
                    "ablated": model.tokenizer.decode(
                        self._generate(ids, layer, patch, max_new_tokens)
                    ),
                }
        return out

    def _check(self, feature: int, layer: int) -> None:
        if not 0 <= layer < NUM_LAYERS:
            raise ValueError(f"layer {layer} out of range 0-{NUM_LAYERS - 1}")
        if not 0 <= feature < D_SAE:
            raise ValueError(f"feature {feature} out of range 0-{D_SAE - 1}")

    def _next_token_probs(
        self, ids: torch.Tensor, layer: int, vec: torch.Tensor | None
    ) -> tuple[torch.Tensor, float]:
        """Next-token distribution after the prompt, optionally steered at `layer`.

        Also returns the mean residual norm at `layer` (excluding <bos>), before
        any steering is added.
        """
        model = self._get_model()
        with torch.no_grad(), model.trace(ids):
            target = model.model.layers[layer]
            resid = target.output.save()
            if vec is not None:
                target.output = target.output + vec
            logits = model.output.logits.save()
        norm = resid[0, 1:].norm(dim=-1).mean().item()
        return logits[0, -1].float().softmax(-1).cpu(), norm

    def _generate(
        self, ids: torch.Tensor, layer: int, edit: Any, max_new_tokens: int
    ) -> list[int]:
        """Greedy completion token ids, editing `layer`'s output on every forward pass.

        `edit` is None, a vector to add, or a callable mapping the residual
        tensor to its replacement.
        """
        model = self._get_model()
        with torch.no_grad(), model.generate(
            ids, max_new_tokens=max_new_tokens, do_sample=False
        ) as tracer:
            if edit is not None:
                # Every forward pass: the prompt, then each generated token.
                with tracer.all():
                    target = model.model.layers[layer]
                    if callable(edit):
                        target.output = edit(target.output)
                    else:
                        target.output = target.output + edit
            out = tracer.result.save()
        return out[0, ids.shape[1]:].tolist()

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
