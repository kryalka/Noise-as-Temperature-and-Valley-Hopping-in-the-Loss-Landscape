from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from ntempvh.eval.metrics import eval_loss_acc
from ntempvh.eval.bn import recalibrate_bn



def lerp_state_dict(sd_a: dict[str, Any], sd_b: dict[str, Any], t: float) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in sd_a.keys():
        a = sd_a[key]
        b = sd_b[key]

        is_bn_buf = ("running_mean" in key) or ("running_var" in key) or ("num_batches_tracked" in key)
        if is_bn_buf or (hasattr(a, "is_floating_point") and (not a.is_floating_point())):
            out[key] = a.clone()
            continue

        out[key] = (1.0 - t) * a + t * b
    return out



def interp_state_dicts_piecewise(
    state_dicts: list[dict[str, Any]],
    t: float,
) -> dict[str, Any]:
    num_segments = len(state_dicts) - 1
    stretched_t = float(np.clip(t, 0.0, 1.0)) * num_segments
    left_idx = int(min(num_segments - 1, max(0, np.floor(stretched_t))))
    local_t = stretched_t - left_idx
    return lerp_state_dict(state_dicts[left_idx], state_dicts[left_idx + 1], float(local_t))



def interp_state_dicts_by_breakpoints(
    state_dicts: list[dict[str, Any]],
    breakpoints: np.ndarray,
    t: float,
) -> dict[str, Any]:
    if len(state_dicts) < 2:
        raise ValueError("Expected at least 2 state dicts for segmented interpolation")
    if len(breakpoints) != len(state_dicts):
        raise ValueError(
            f"Breakpoints/state-dict length mismatch: {len(breakpoints)} vs {len(state_dicts)}"
        )

    clipped_t = float(np.clip(t, 0.0, 1.0))
    if clipped_t <= 0.0:
        return lerp_state_dict(state_dicts[0], state_dicts[1], 0.0)
    if clipped_t >= 1.0:
        return lerp_state_dict(state_dicts[-2], state_dicts[-1], 1.0)

    left_idx = int(np.searchsorted(breakpoints, clipped_t, side="right") - 1)
    left_idx = min(max(left_idx, 0), len(state_dicts) - 2)

    left_t = float(breakpoints[left_idx])
    right_t = float(breakpoints[left_idx + 1])
    width = max(right_t - left_t, 1e-12)
    local_t = (clipped_t - left_t) / width
    return lerp_state_dict(state_dicts[left_idx], state_dicts[left_idx + 1], float(local_t))


@torch.no_grad()
def eval_model(model: nn.Module, loader, device: torch.device) -> tuple[float, float]:
    return eval_loss_acc(model, loader, device)



@torch.no_grad()
def eval_endpoint_state_dict(
    *,
    model_factory,
    model_name: str,
    num_classes: int,
    state_dict: dict[str, Any],
    bn_loader,
    eval_loader,
    device: torch.device,
    bn_batches: int,
) -> tuple[float, float]:
    endpoint_model = model_factory(model_name, num_classes=num_classes).to(device)
    endpoint_model.load_state_dict(state_dict, strict=True)
    recalibrate_bn(endpoint_model, bn_loader, device, num_batches=bn_batches, reset_stats=True)
    return eval_model(endpoint_model, eval_loader, device)
