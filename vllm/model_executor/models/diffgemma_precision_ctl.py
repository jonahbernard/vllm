"""Per-token mixed-precision controller for DiffusionGemma (experiment, env-gated).

Lets the native AITER MXFP4 MoE run *some canvas positions through the native
W4A4 kernel and others through a full-bf16 expert GEMM*. The fp4 path uses the
quantized weights with fp4 activations; the bf16 path uses the TRUE bf16 expert
weights loaded from the original (un-quantized) DiffusionGemma card with bf16
activations. Routing is identical for both; only the expert-GEMM precision
differs. The MoE method runs both paths and blends their outputs per row with a
``torch.where`` driven by the mask stashed here, reproducing a per-token mixed-
precision MoE.

Selection policy for this experiment: a canvas position that is *already
committed* (per-position confidence above the sampler's threshold, measured from
the previous denoising step) uses the native W4A4 path; a not-yet-committed
position uses the full-bf16 path. The sampler computes the committed mask each
step and stashes it here; the MoE reads it on the next forward (committed-ness is
layer-independent, so the same mask applies to every MoE layer in that forward).

Nothing here runs unless ``VLLM_DIFFGEMMA_MIXED_PREC`` is set, OR one of the
per-site precision overrides below is set::

    VLLM_DIFFGEMMA_MIXED_PREC=committed vllm serve ... \
        --moe-backend aiter --enforce-eager

Per-token modes (``VLLM_DIFFGEMMA_MIXED_PREC``):
  - ``committed``  committed positions -> W4A4, uncommitted -> bf16 (request).
  - ``invert``     committed -> bf16, uncommitted -> W4A4 (control / sanity).
  - ``bf16``       ALL positions -> bf16, on every forward (verification: output
                   should match the pure bf16 server within bf16 tolerance).

Per-site precision overrides (independent of the per-token mask above): force a
whole *layer type* to one precision, so the MoE expert GEMMs and the non-MoE
dense MLP GEMMs can be set separately::

    VLLM_DIFFGEMMA_MOE_PREC=mxfp4  VLLM_DIFFGEMMA_MLP_PREC=bf16  vllm serve ...
    VLLM_DIFFGEMMA_MOE_PREC=bf16   VLLM_DIFFGEMMA_MLP_PREC=mxfp4 vllm serve ...

Each var takes ``bf16`` (all rows -> true bf16 weights) or ``mxfp4`` (all rows ->
native W4A4). When a site's override is set it wins over the per-token mode for
that site; when it is unset that site falls back to the per-token mask. Either
override alone is enough to enable the controller (no ``MIXED_PREC`` needed).

This is a SEPARATE flag from both probes. It mutates numerics (it changes which
kernel each row runs through), so it is an experiment switch, not an observer.
Run with ``--enforce-eager`` so the second kernel is dispatched inline.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_CTL: "DiffgemmaPrecisionCtl | None" = None
_INITIALIZED = False


def get_precision_ctl() -> "DiffgemmaPrecisionCtl | None":
    """Return the singleton controller, or None when disabled.

    Lazily reads ``VLLM_DIFFGEMMA_MIXED_PREC`` and the per-site overrides
    ``VLLM_DIFFGEMMA_MOE_PREC`` / ``VLLM_DIFFGEMMA_MLP_PREC`` once. When none are
    set, returns None on every call so the hot path pays only a single attribute
    load.
    """
    global _CTL, _INITIALIZED
    if _INITIALIZED:
        return _CTL
    _INITIALIZED = True
    mode = os.environ.get("VLLM_DIFFGEMMA_MIXED_PREC")
    moe_prec = os.environ.get("VLLM_DIFFGEMMA_MOE_PREC")
    mlp_prec = os.environ.get("VLLM_DIFFGEMMA_MLP_PREC")
    if mode or moe_prec or mlp_prec:
        _CTL = DiffgemmaPrecisionCtl(
            (mode or "committed").strip().lower(),
            site_prec={
                "moe": moe_prec.strip().lower() if moe_prec else None,
                "mlp": mlp_prec.strip().lower() if mlp_prec else None,
            },
        )
    return _CTL


class DiffgemmaPrecisionCtl:
    """Holds the per-row bf16 mask for the current forward.

    The sampler calls ``set_committed`` after each step with the per-position
    committed mask; the emulation MoE calls ``bf16_row_mask`` during the next
    forward to decide which rows keep true bf16 activations.
    """

    def __init__(self, mode: str, site_prec: dict[str, str | None] | None = None):
        if mode not in ("committed", "invert", "bf16"):
            raise ValueError(
                f"VLLM_DIFFGEMMA_MIXED_PREC must be 'committed', 'invert', or "
                f"'bf16', got {mode!r}"
            )
        self.mode = mode
        # Per-site precision overrides: {"moe": "bf16"|"mxfp4"|None,
        # "mlp": ...}. When a site is set it forces that whole layer type to one
        # precision, ignoring the per-token committed mask for that site.
        self.site_prec: dict[str, str | None] = site_prec or {}
        for site, prec in self.site_prec.items():
            if prec is not None and prec not in ("bf16", "mxfp4"):
                raise ValueError(
                    f"VLLM_DIFFGEMMA_{site.upper()}_PREC must be 'bf16' or "
                    f"'mxfp4', got {prec!r}"
                )
        # [num_decode, CL] bool: True where the position is committed. Updated by
        # the sampler each step; consumed by the next forward's MoE layers.
        self._committed: torch.Tensor | None = None
        logger.warning(
            "DiffgemmaPrecisionCtl ENABLED (mode=%s, site_prec=%s). Run with "
            "--moe-backend aiter --enforce-eager.",
            mode,
            self.site_prec,
        )

    def reset(self) -> None:
        """Drop any stored mask so the next forward falls back to pure fp4.

        Called at request boundaries (prefill) so a brand-new request's first
        decode forward does not inherit the *previous* request's committed mask.
        Without this, decode step 0 of every request would run with a stale,
        foreign commit pattern before its own sampler has produced a real mask.
        """
        self._committed = None

    def set_committed(self, committed: torch.Tensor) -> None:
        """Store the per-position committed mask for the next forward.

        Args:
            committed: bool tensor ``[num_decode, CL]``, True where that canvas
                position is committed (confidence above threshold).
        """
        self._committed = committed.detach().to(torch.bool)

    def bf16_row_mask(
        self, n_rows: int, device: torch.device, site: str | None = None
    ) -> torch.Tensor | None:
        """Per-row mask (True = keep bf16 activations) for an ``[n_rows, H]`` input.

        ``site`` identifies the call site (``"moe"`` for the expert GEMMs,
        ``"mlp"`` for the non-MoE dense MLP GEMMs). When that site has a
        precision override set (``VLLM_DIFFGEMMA_{MOE,MLP}_PREC``), it wins over
        the per-token mode: ``bf16`` -> all rows bf16 (all-ones mask), ``mxfp4``
        -> all rows fp4 (None, so the caller runs the native W4A4 path). This
        applies on every forward, independent of the committed mask, so MoE and
        MLP precision can be chosen separately.

        With no override for this site, falls back to the per-token mask: the
        stored mask is ``[num_decode, CL]`` in row-major (request-outer,
        position-inner) order, which is exactly the token order of a decode
        forward's MoE input (request r's canvas position p is row r*CL+p). The
        mask therefore applies ONLY to a forward whose row count equals the
        mask's element count. Any other forward (a prefill, whose rows are prompt
        tokens; or a decode whose request set changed shape) does NOT correspond
        to this mask row-for-row, so we return None and the caller falls back to
        pure fp4 rather than blending against a misaligned mask.

        Returns None when no mask is stored yet (first decode forward of a
        request, or just after ``reset``), in which case the caller runs the
        ordinary native W4A4 path.

        In ``bf16`` mode every row goes through the bf16 path on every forward
        (including prefill), independent of any stored committed mask, so the
        whole model runs in bf16 — a verification knob whose output should match
        the pure bf16 server within bf16 tolerance.
        """
        site_prec = self.site_prec.get(site) if site is not None else None
        if site_prec == "bf16":
            return torch.ones(n_rows, dtype=torch.bool, device=device)
        if site_prec == "mxfp4":
            return None
        if self.mode == "bf16":
            return torch.ones(n_rows, dtype=torch.bool, device=device)
        if self._committed is None:
            return None
        committed_flat = self._committed.reshape(-1).to(device)
        # Exact-match only: the mask is row-for-row meaningful for this forward
        # iff its element count equals the MoE row count. A mismatch means this
        # is a different forward (prefill, or a changed request batch) than the
        # one the mask was computed for; tiling/trimming would silently apply a
        # foreign commit pattern, so refuse and fall back to pure fp4.
        if committed_flat.numel() != n_rows:
            return None
        # committed -> fp4 (QDQ), uncommitted -> bf16. bf16 mask = ~committed.
        bf16 = ~committed_flat
        if self.mode == "invert":
            bf16 = ~bf16
        return bf16
