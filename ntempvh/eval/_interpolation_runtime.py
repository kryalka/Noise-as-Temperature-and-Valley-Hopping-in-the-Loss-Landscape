from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ntempvh.eval._interpolation_core import (
    eval_endpoint_state_dict,
    eval_model,
    interp_state_dicts_by_breakpoints,
    interp_state_dicts_piecewise,
    lerp_state_dict,
)
from ntempvh.eval._interpolation_paths import resolve_path_sequence
from ntempvh.eval._interpolation_runtime_steps import (
    build_checkpoint_bundle,
    build_interpolation_meta,
    build_loader_bundle,
    build_runtime_settings,
    collect_interpolation_rows,
)



def interpolate_state_dict(
    *,
    path_type: str,
    state_dicts: list[dict[str, Any]],
    t: float,
    segment_breakpoints: np.ndarray | None,
) -> dict[str, Any]:
    if path_type == "linear":
        return lerp_state_dict(state_dicts[0], state_dicts[-1], t)
    if path_type == "observed":
        if segment_breakpoints is None:
            raise RuntimeError("Observed path interpolation requires segment breakpoints")
        return interp_state_dicts_by_breakpoints(state_dicts, segment_breakpoints, t)
    if path_type in ("polyline", "piecewise", "piecewise_linear"):
        return interp_state_dicts_piecewise(state_dicts, t)
    raise ValueError(f"Unknown path.type: {path_type}")



def run_interpolation_compute(
    *,
    ckpt_a: str,
    ckpt_b: str,
    cfg: dict[str, Any],
    out_dir: str,
    build_artifact_context_fn,
    get_device_fn,
    set_seed_fn,
    validate_checkpoint_pair_fn,
    get_num_classes_fn,
    select_train_loader_fn,
    select_test_loader_fn,
    make_model_fn,
    ensure_dir_fn,
    save_json_fn,
) -> Path:
    artifact = build_artifact_context_fn(ckpt_a, ckpt_b, cfg)
    settings = build_runtime_settings(cfg=cfg, artifact=artifact, get_device_fn=get_device_fn)
    out_path = ensure_dir_fn(out_dir)

    checkpoint_bundle = build_checkpoint_bundle(
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        settings=settings,
        set_seed_fn=set_seed_fn,
        validate_checkpoint_pair_fn=validate_checkpoint_pair_fn,
        get_num_classes_fn=get_num_classes_fn,
        resolve_path_sequence_fn=resolve_path_sequence,
    )
    loader_bundle = build_loader_bundle(
        settings=settings,
        dataset_name=checkpoint_bundle["dataset_name"],
        select_train_loader_fn=select_train_loader_fn,
        select_test_loader_fn=select_test_loader_fn,
    )

    endpoint_a_loss, endpoint_a_acc = eval_endpoint_state_dict(
        model_factory=make_model_fn,
        model_name=checkpoint_bundle["model_name"],
        num_classes=checkpoint_bundle["num_classes"],
        state_dict=checkpoint_bundle["state_dicts"][0],
        bn_loader=loader_bundle["bn_loader"],
        eval_loader=loader_bundle["eval_loader"],
        device=settings.device,
        bn_batches=settings.bn_batches,
    )
    endpoint_b_loss, endpoint_b_acc = eval_endpoint_state_dict(
        model_factory=make_model_fn,
        model_name=checkpoint_bundle["model_name"],
        num_classes=checkpoint_bundle["num_classes"],
        state_dict=checkpoint_bundle["state_dicts"][-1],
        bn_loader=loader_bundle["bn_loader"],
        eval_loader=loader_bundle["eval_loader"],
        device=settings.device,
        bn_batches=settings.bn_batches,
    )

    model = make_model_fn(
        checkpoint_bundle["model_name"],
        num_classes=checkpoint_bundle["num_classes"],
    ).to(settings.device)
    print(
        f"interpolation path_type={settings.path_type}, num_points={settings.num_points}, "
        f"split={settings.eval_split}, bn_batches={settings.bn_batches}"
    )
    arr = collect_interpolation_rows(
        model=model,
        state_dicts=checkpoint_bundle["state_dicts"],
        path_type=settings.path_type,
        num_points=settings.num_points,
        segment_breakpoints=checkpoint_bundle["segment_breakpoints"],
        interpolate_state_dict_fn=interpolate_state_dict,
        bn_loader=loader_bundle["bn_loader"],
        eval_loader=loader_bundle["eval_loader"],
        device=settings.device,
        bn_batches=settings.bn_batches,
        eval_model_fn=eval_model,
    )
    arr[0, 1] = float(endpoint_a_loss)
    arr[0, 2] = float(endpoint_a_acc)
    arr[-1, 1] = float(endpoint_b_loss)
    arr[-1, 2] = float(endpoint_b_acc)

    out_file = Path(out_path) / f"{artifact['stem']}.csv"
    np.savetxt(out_file, arr, delimiter=",", header="t,val_loss,val_acc", comments="")

    meta = build_interpolation_meta(
        settings=settings,
        ckpt_a=ckpt_a,
        ckpt_b=ckpt_b,
        model_name=checkpoint_bundle["model_name"],
        dataset_name=checkpoint_bundle["dataset_name"],
        path_meta=checkpoint_bundle["path_meta"],
        endpoint_a_loss=endpoint_a_loss,
        endpoint_a_acc=endpoint_a_acc,
        endpoint_b_loss=endpoint_b_loss,
        endpoint_b_acc=endpoint_b_acc,
        checkpoint_a=checkpoint_bundle["checkpoint_a"],
        checkpoint_b=checkpoint_bundle["checkpoint_b"],
    )
    save_json_fn(Path(out_file).with_suffix(".meta.json"), meta)
    return out_file
