"""Confidence probe for DiffusionGemma denoising.

Env-gated, zero-overhead-when-off recorder that dumps per-token, per-denoising
-step confidence signals so the MXFP4 (W4A4) path can be A/B'd against the BF16
path. Nothing here runs unless ``VLLM_DIFFGEMMA_PROBE`` is set.

Enable by pointing the env var at an output JSONL path::

    VLLM_DIFFGEMMA_PROBE=/path/to/run.jsonl vllm serve ...

Each emitted line is one decode request at one denoising step::

    {
      "tag": "mxfp4",          # VLLM_DIFFGEMMA_PROBE_TAG (defaults to "run")
      "slot": 0,               # request slot index (stable for a single req)
      "req_id": "cmpl-abc-0",  # OpenAI request id when available
      "step": 7,               # denoising step counter for this slot
      "committing": false,     # was this the commit (emit) step?
      "mean_entropy": 1.83,    # per-request mean token entropy (the confidence
                               #   signal the sampler thresholds on)
      "confidence_threshold": 2.5,
      "confident": true,       # mean_entropy < confidence_threshold
      "n_valid": 32,           # real (non-padded) canvas positions
      "token_entropy": [...],  # per-position entropy, length n_valid
      "max_prob": [...],       # per-position top-1 probability, length n_valid
      "argmax_token": [...]    # per-position argmax token id, length n_valid
    }

The probe reads the same temperature-scaled logits the sampler uses, so the
recorded entropy/confidence is exactly what drives convergence — not an
approximation. Recording happens outside the compiled region, so it never
perturbs the sampler's own math.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

_PROBE: "DiffgemmaProbe | None" = None
_INITIALIZED = False


def get_probe() -> "DiffgemmaProbe | None":
    """Return the singleton probe, or None when probing is disabled.

    Lazily reads ``VLLM_DIFFGEMMA_PROBE`` once. When unset, returns None on
    every call so the sampler hot path pays only a single attribute load.
    """
    global _PROBE, _INITIALIZED
    if _INITIALIZED:
        return _PROBE
    _INITIALIZED = True
    path = os.environ.get("VLLM_DIFFGEMMA_PROBE")
    if path:
        tag = os.environ.get("VLLM_DIFFGEMMA_PROBE_TAG", "run")
        _PROBE = DiffgemmaProbe(path, tag)
    return _PROBE


class DiffgemmaProbe:
    """Append-only JSONL recorder for per-step denoising confidence."""

    def __init__(self, path: str, tag: str):
        self.path = path
        self.tag = tag
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Truncate any prior run so each server launch starts a clean file.
        self._fh = open(path, "w", buffering=1)

    def record_step(
        self,
        *,
        scaled: torch.Tensor,
        decode_slots: torch.Tensor,
        step_tensor: torch.Tensor,
        is_committing: torch.Tensor,
        valid_canvas_len_np: Any,
        confidence_threshold: float,
        req_ids_by_slot: dict[int, str] | None = None,
    ) -> None:
        """Dump one JSONL line per decode request for the current step.

        Args:
            scaled: ``[num_decode, CL, vocab]`` temperature-scaled logits
                returned by ``_compiled_sample_step`` (already fp32).
            decode_slots: ``[num_decode]`` slot index per decode request.
            step_tensor: per-slot denoising step counter. Read AFTER the
                compiled step (so it is the post-increment value).
            is_committing: ``[num_decode]`` bool, True for the commit step.
            valid_canvas_len_np: real canvas length per decode request.
            confidence_threshold: sampler's mean-entropy commit threshold.
            req_ids_by_slot: optional slot -> request id map for alignment.
        """
        num_decode = scaled.shape[0]
        if num_decode == 0:
            return

        log_probs = scaled.log_softmax(dim=-1)
        probs = log_probs.exp()
        token_entropy = -(probs * log_probs).sum(dim=-1)  # [num_decode, CL]
        max_prob = probs.max(dim=-1).values  # [num_decode, CL]
        argmax_token = scaled.argmax(dim=-1)  # [num_decode, CL]

        slots = decode_slots.tolist()
        steps = step_tensor[decode_slots].tolist()
        committing = is_committing.tolist()
        ent_cpu = token_entropy.cpu()
        maxp_cpu = max_prob.cpu()
        argmax_cpu = argmax_token.cpu()
        valid = list(valid_canvas_len_np)

        for i in range(num_decode):
            n = int(valid[i])
            ent_i = ent_cpu[i, :n]
            mean_entropy = float(ent_i.mean())
            slot = int(slots[i])
            rec = {
                "tag": self.tag,
                "slot": slot,
                "req_id": (req_ids_by_slot or {}).get(slot),
                "step": int(steps[i]),
                "committing": bool(committing[i]),
                "mean_entropy": mean_entropy,
                "confidence_threshold": confidence_threshold,
                "confident": mean_entropy < confidence_threshold,
                "n_valid": n,
                "token_entropy": [round(x, 5) for x in ent_i.tolist()],
                "max_prob": [round(x, 5) for x in maxp_cpu[i, :n].tolist()],
                "argmax_token": argmax_cpu[i, :n].tolist(),
            }
            self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
