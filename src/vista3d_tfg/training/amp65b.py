"""Retry the same patch after AMP overflow; never count a skipped update."""
from __future__ import annotations

import math


MAX_RETRIES = 8


def optimizer_step(model, optimizer, scaler, loss_fn, x, y, max_norm, device,
                   *, on_retry=None):
    import torch

    parameters = list(model.parameters())
    for attempt in range(MAX_RETRIES + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        if not bool(torch.isfinite(loss)):
            raise ValueError("Non-finite training loss; no optimizer update.")
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grads = [p.grad for p in parameters if p.grad is not None]
        if not grads:
            raise ValueError("Missing training gradients.")
        finite = bool(torch.stack([torch.isfinite(g).all() for g in grads]).all())
        if not finite:
            if not scaler.is_enabled():
                raise ValueError("Non-finite FP32 gradients; no optimizer update.")
            scale_before = float(scaler.get_scale())
            # unscale_ recorded the non-finite gradients. GradScaler skips
            # optimizer.step and resets its growth tracker on update().
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            event = {"retry": attempt + 1, "scale_before": scale_before,
                     "scale_after": scale_after, "optimizer_update": False}
            if on_retry is not None:
                on_retry(event)
            if not math.isfinite(scale_after) or not 0 < scale_after < scale_before:
                raise ValueError("AMP backoff failed; no optimizer update.")
            if attempt == MAX_RETRIES:
                raise ValueError("Non-finite gradients persist after 8 AMP retries; epoch not committed.")
            del logits, loss, grads
            # x/y are the SAME augmented tensors. The frozen model has no
            # dropout or running normalization statistics; no case is skipped.
            continue
        norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)
        if float(norm) <= 0:
            raise ValueError("Zero training gradient.")
        scaler.step(optimizer)
        scaler.update()
        return float(loss.detach()), attempt
    raise AssertionError("Unreachable AMP retry state")
