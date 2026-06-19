"""Pre-GEMM activation-statistics probe for DiffusionGemma.

Env-gated, zero-overhead-when-off recorder that dumps per-GEMM, per-layer,
per-denoising-step activation statistics so the MXFP4 (W4A4) path can be A/B'd
against the BF16 path at the level of the tensors fp4 actually quantizes — the
activations *feeding into* each GEMM. The signal of interest is outliers
(outlier tokens, outlier channels, inflated per-block absmax), since fp4 block
scaling keys off per-block absmax and a single outlier inflates the scale for
its whole block.

This is a SEPARATE flag from the confidence probe (``VLLM_DIFFGEMMA_PROBE``);
both may be enabled in the same run, each writing its own file. Nothing here
runs unless ``VLLM_DIFFGEMMA_ACT_PROBE`` is set::

    VLLM_DIFFGEMMA_ACT_PROBE=/path/to/act.jsonl vllm serve ... --enforce-eager

IMPORTANT: capture is inline Python and only runs in eager mode. Run the server
with ``--enforce-eager`` when this probe is enabled, otherwise FULL CUDA-graph
replay will bypass recording. (The YOCO ``@support_torch_compile`` decoders are
already no-ops unless ``kv_sharing_fast_prefill`` is set.)

Optional tuning env vars:
  - ``VLLM_DIFFGEMMA_ACT_PROBE_TAG``     run tag (default "run")
  - ``VLLM_DIFFGEMMA_ACT_PROBE_BLOCK``   block size for block-absmax (default 32)
  - ``VLLM_DIFFGEMMA_ACT_PROBE_OUTLIER_K`` std multiple for frac_large (default 6)

Output is JSONL. Two kinds of line:

1. One activation record per (layer, gemm site, forward)::

    {
      "kind": "act",
      "tag": "mxfp4",
      "forward_id": 41,       # monotonic per model forward; join key to ctx line
      "site": "mlp.gate_up",  # which GEMM input this is
      "layer": 7,
      "n_tokens": 256,        # rows T
      "hidden": 2560,         # cols H (this GEMM's input dim)
      "block": 32,
      "absmax": ..., "mean": ..., "std": ..., "rms": ...,
      "frac_large": ...,      # fraction of |a| > k*std
      "tok_l2": [...],        # length T
      "tok_absmax": [...],    # length T
      "tok_kurtosis": [...],  # length T
      "ch_absmax": [...],     # length H
      "ch_rms": [...],        # length H
      "ch_mean": [...],       # length H
      "blk_absmax_max": ..., "blk_absmax_p99": ...,
      "blk_absmax_p50": ..., "blk_absmax_mean": ...,
      "blk_trimmed": 0        # channels dropped when H % block != 0
    }

2. One context record per model forward, mapping forward_id -> denoising step /
   slot metadata (emitted from the diffusion runner, where that info lives)::

    {
      "kind": "ctx",
      "tag": "mxfp4",
      "forward_id": 41,
      "step_by_slot": {"1": 7},      # denoising step per request slot
      "valid_by_slot": {"1": 256},   # real (non-padded) canvas length per slot
      "req_id_by_slot": {"1": "cmpl-abc"}
    }

Analysis joins act records to ctx records on ``forward_id`` to recover which
denoising step / position each statistic belongs to.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_PROBE: "DiffgemmaActProbe | None" = None
_INITIALIZED = False


def get_act_probe() -> "DiffgemmaActProbe | None":
    """Return the singleton activation probe, or None when disabled.

    Lazily reads ``VLLM_DIFFGEMMA_ACT_PROBE`` once. When unset, returns None on
    every call so the model hot path pays only a single attribute load.
    """
    global _PROBE, _INITIALIZED
    if _INITIALIZED:
        return _PROBE
    _INITIALIZED = True
    path = os.environ.get("VLLM_DIFFGEMMA_ACT_PROBE")
    if path:
        tag = os.environ.get("VLLM_DIFFGEMMA_ACT_PROBE_TAG", "run")
        block = int(os.environ.get("VLLM_DIFFGEMMA_ACT_PROBE_BLOCK", "32"))
        outlier_k = float(os.environ.get("VLLM_DIFFGEMMA_ACT_PROBE_OUTLIER_K", "6"))
        _PROBE = DiffgemmaActProbe(path, tag, block, outlier_k)
    return _PROBE


def _round_list(t: torch.Tensor) -> list[float]:
    return [round(x, 5) for x in t.tolist()]


class DiffgemmaActProbe:
    """Append-only JSONL recorder for pre-GEMM activation statistics."""

    def __init__(self, path: str, tag: str, block: int, outlier_k: float):
        self.path = path
        self.tag = tag
        self.block = block
        self.outlier_k = outlier_k
        # Monotonic id incremented at the start of every model forward; stamped
        # onto each activation record so analysis can join them to the per-forward
        # context line emitted by the runner.
        self.forward_id = -1
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        # Truncate any prior run so each server launch starts a clean file.
        self._fh = open(path, "w", buffering=1)
        logger.warning(
            "DiffgemmaActProbe ENABLED -> %s (tag=%s, block=%d, outlier_k=%g). "
            "Run with --enforce-eager; inline capture is skipped under CUDA "
            "graph replay.",
            path,
            tag,
            block,
            outlier_k,
        )

    def begin_forward(self) -> None:
        """Advance the forward counter. Call once at the start of each model
        forward, before any layer runs."""
        self.forward_id += 1

    def record(self, site: str, layer_idx: int, x: torch.Tensor) -> None:
        """Compute and append activation statistics for one pre-GEMM tensor.

        Args:
            site: which GEMM input this is (e.g. "qkv_proj", "mlp.gate_up").
            layer_idx: decoder layer index.
            x: the activation feeding the GEMM, shape ``[..., H]``. Flattened to
               ``[T, H]`` (T = total tokens/rows across the leading dims).
        """
        if x is None or x.numel() == 0:
            return
        # Flatten any leading dims to a single token axis; last dim is channels.
        a = x.detach().reshape(-1, x.shape[-1]).float()
        T, H = a.shape

        absA = a.abs()
        std = a.std()
        rms = a.pow(2).mean().sqrt()

        # Per-token (rows) — find outlier tokens.
        tok_l2 = a.norm(dim=1)
        tok_absmax = absA.max(dim=1).values
        mu_t = a.mean(dim=1, keepdim=True)
        diff = a - mu_t
        var_t = diff.pow(2).mean(dim=1)
        m4_t = diff.pow(4).mean(dim=1)
        tok_kurtosis = m4_t / (var_t.pow(2) + 1e-12)

        # Per-channel (cols) — find outlier channels.
        ch_absmax = absA.max(dim=0).values
        ch_rms = a.pow(2).mean(dim=0).sqrt()
        ch_mean = a.mean(dim=0)

        # Per-block absmax distribution: blocks of `block` channels per token row,
        # mirroring how mxfp4 assigns one fp4 scale per contiguous channel block.
        B = self.block
        nb = H // B
        if nb > 0:
            blocks = a[:, : nb * B].reshape(T, nb, B)
            blk_absmax = blocks.abs().amax(dim=2).reshape(-1)  # [T*nb]
            blk_max = float(blk_absmax.max())
            blk_p99 = float(blk_absmax.quantile(0.99))
            blk_p50 = float(blk_absmax.median())
            blk_mean = float(blk_absmax.mean())
            blk_trimmed = H - nb * B
        else:
            blk_max = blk_p99 = blk_p50 = blk_mean = float("nan")
            blk_trimmed = H

        rec = {
            "kind": "act",
            "tag": self.tag,
            "forward_id": self.forward_id,
            "site": site,
            "layer": int(layer_idx),
            "n_tokens": T,
            "hidden": H,
            "block": B,
            "absmax": round(float(absA.max()), 5),
            "mean": round(float(a.mean()), 5),
            "std": round(float(std), 5),
            "rms": round(float(rms), 5),
            "frac_large": round(
                float((absA > self.outlier_k * std).float().mean()), 8
            ),
            "tok_l2": _round_list(tok_l2.cpu()),
            "tok_absmax": _round_list(tok_absmax.cpu()),
            "tok_kurtosis": _round_list(tok_kurtosis.cpu()),
            "ch_absmax": _round_list(ch_absmax.cpu()),
            "ch_rms": _round_list(ch_rms.cpu()),
            "ch_mean": _round_list(ch_mean.cpu()),
            "blk_absmax_max": round(blk_max, 5),
            "blk_absmax_p99": round(blk_p99, 5),
            "blk_absmax_p50": round(blk_p50, 5),
            "blk_absmax_mean": round(blk_mean, 5),
            "blk_trimmed": int(blk_trimmed),
        }
        self._fh.write(json.dumps(rec) + "\n")

    def record_context(
        self,
        *,
        step_by_slot: dict[int, int],
        valid_by_slot: dict[int, int],
        req_id_by_slot: dict[int, Any] | None = None,
    ) -> None:
        """Emit the forward_id -> step/slot mapping for the current forward."""
        rec = {
            "kind": "ctx",
            "tag": self.tag,
            "forward_id": self.forward_id,
            "step_by_slot": {str(k): int(v) for k, v in step_by_slot.items()},
            "valid_by_slot": {str(k): int(v) for k, v in valid_by_slot.items()},
            "req_id_by_slot": {
                str(k): v for k, v in (req_id_by_slot or {}).items()
            },
        }
        self._fh.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass
