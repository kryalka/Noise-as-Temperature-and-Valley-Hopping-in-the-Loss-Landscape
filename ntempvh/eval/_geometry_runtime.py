from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm



@torch.no_grad()
def run_geometry_compute(
    ckpt_path: str,
    geometry_cfg_path: str,
    out_path: str,
    *,
    load_yaml_fn,
    build_geometry_artifact_context_fn,
    get_device_fn,
    set_seed_fn,
    get_num_classes_fn,
    select_train_loader_fn,
    call_with_supported_kwargs_fn,
    make_model_fn,
    eval_classification_fn,
    recalibrate_bn_fn,
    params_to_vector_fn,
    vector_to_params_fn,
    sample_unit_directions_fn,
    save_failure_json_fn,
    ensure_dir_fn,
    save_json_fn,
) -> Path:
    cfg = load_yaml_fn(geometry_cfg_path)
    success_artifact = build_geometry_artifact_context_fn(ckpt_path, cfg, failed=False)
    failure_artifact = build_geometry_artifact_context_fn(ckpt_path, cfg, failed=True)
    geometry_cfg = success_artifact["config"]["geometry"]

    alpha = float(geometry_cfg["alpha"])
    num_directions = int(geometry_cfg["num_directions"])
    eval_batch_size = int(geometry_cfg["eval_batch_size"])
    bn_batches = int(geometry_cfg["bn_recalib_batches"])
    num_eval_batches = geometry_cfg["num_eval_batches"]
    device = get_device_fn()

    ckpt = torch.load(ckpt_path, map_location="cpu")
    set_seed_fn(int(ckpt.get("seed", 0)))
    model_name = str(ckpt["model"]).lower()
    dataset_name = str(ckpt.get("dataset", "")).lower()
    num_classes = int(get_num_classes_fn(dataset_name))
    data_root = str(success_artifact["config"]["data_root"])
    data_cfg = dict((cfg.get("data", {}) if isinstance(cfg, dict) else {}) or {})
    train_loader_fn = select_train_loader_fn(dataset_name)

    loaders = call_with_supported_kwargs_fn(
        train_loader_fn,
        root=data_root,
        batch_size=eval_batch_size,
        val_batch_size=eval_batch_size,
        val_size=int(success_artifact["config"]["evaluation"]["val_size"]),
        split_seed=int(success_artifact["config"]["evaluation"]["split_seed"]),
        shuffle_seed=int(success_artifact["config"]["evaluation"]["split_seed"]),
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", True)),
    )
    val_loader = loaders.val
    bn_loader = loaders.bn

    model = make_model_fn(model_name, num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()

    raw_base = eval_classification_fn(model, val_loader, device, max_batches=num_eval_batches)
    recalibrate_bn_fn(model, bn_loader, device, num_batches=bn_batches, reset_stats=False)
    base = eval_classification_fn(model, val_loader, device, max_batches=num_eval_batches)
    base_loss = float(base["loss"])
    base_acc = float(base["acc"])

    if not np.isfinite(base_loss) or not np.isfinite(base_acc):
        return save_failure_json_fn(
            out_dir=out_path, artifact=failure_artifact, ckpt_path=ckpt_path, model_name=model_name,
            dataset_name=dataset_name, device=device, alpha=alpha, m=num_directions,
            eval_batch_size=eval_batch_size, num_eval_batches=num_eval_batches, bn_batches=bn_batches,
            raw_base=raw_base, bn_base=base, reason="non_finite_bn_base",
        )
    if base_acc < 0.2 or base_loss > 2.5:
        return save_failure_json_fn(
            out_dir=out_path, artifact=failure_artifact, ckpt_path=ckpt_path, model_name=model_name,
            dataset_name=dataset_name, device=device, alpha=alpha, m=num_directions,
            eval_batch_size=eval_batch_size, num_eval_batches=num_eval_batches, bn_batches=bn_batches,
            raw_base=raw_base, bn_base=base, reason="unstable_bn_recalibration",
            extra={"threshold_acc_min": 0.2, "threshold_loss_max": 2.5},
        )

    theta0 = params_to_vector_fn(model).detach().to(device)
    theta_norm = float(torch.norm(theta0).item())
    epsilon = float(alpha * theta_norm)
    if not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError(f"Bad epsilon: eps={epsilon}, alpha={alpha}, ||theta||={theta_norm}")

    directions = sample_unit_directions_fn(int(theta0.numel()), num_directions, device=device, dtype=theta0.dtype)
    per_direction: list[float] = []
    start_total = time.time()
    pbar = tqdm(range(num_directions), desc="geometry directions")
    for idx in pbar:
        dir_start = time.time()
        direction = directions[idx]
        theta_plus = theta0 + epsilon * direction
        vector_to_params_fn(model, theta_plus)
        recalibrate_bn_fn(model, bn_loader, device, num_batches=bn_batches, reset_stats=False)
        loss_plus = float(eval_classification_fn(model, val_loader, device, max_batches=num_eval_batches)["loss"])

        theta_minus = theta0 - epsilon * direction
        vector_to_params_fn(model, theta_minus)
        recalibrate_bn_fn(model, bn_loader, device, num_batches=bn_batches, reset_stats=False)
        loss_minus = float(eval_classification_fn(model, val_loader, device, max_batches=num_eval_batches)["loss"])
        vector_to_params_fn(model, theta0)

        sec = (loss_plus + loss_minus - 2.0 * base_loss) / (epsilon * epsilon)
        per_direction.append(float(sec))
        elapsed_total = time.time() - start_total
        eta = (elapsed_total / (idx + 1)) * (num_directions - idx - 1)
        pbar.set_postfix(dir_sec=f"{time.time() - dir_start:.1f}s", eta_min=f"{eta / 60:.1f}", last_sec=f"{sec:.3e}")

    kappa_tr = float(np.mean(per_direction)) if per_direction else float("nan")
    kappa_std = float(np.std(per_direction, ddof=1)) if len(per_direction) > 1 else 0.0
    out_dir = ensure_dir_fn(out_path)
    json_path = Path(out_dir) / f"{success_artifact['stem']}.json"
    out = {
        "ckpt": str(ckpt_path),
        "dataset": dataset_name,
        "model": model_name,
        "device": str(device),
        "alpha": alpha,
        "num_directions": num_directions,
        "eval_batch_size": eval_batch_size,
        "num_eval_batches": num_eval_batches,
        "bn_recalib_batches": bn_batches,
        "theta_norm": theta_norm,
        "epsilon": epsilon,
        "base": base,
        "kappa_tr": kappa_tr,
        "kappa_tr_std": kappa_std,
        "sigma_kappa": float(kappa_std),
        "anisotropy": float(kappa_std),
        "per_direction": per_direction,
        "artifact": {
            "kind": "geometry",
            "checkpoint_tag": success_artifact["checkpoint_tag"],
            "config_signature": success_artifact["config_signature"],
            "stem": success_artifact["stem"],
        },
    }
    save_json_fn(json_path, out)

    csv_path = Path(out_dir) / "geometries.csv"
    header = "ckpt,model,dataset,alpha,num_directions,eval_batch_size,num_eval_batches,theta_norm,epsilon,base_loss,base_acc,kappa_tr,kappa_tr_std,sigma_kappa,anisotropy"
    row = [
        str(ckpt_path), model_name, dataset_name, f"{alpha:.10g}", str(num_directions), str(eval_batch_size),
        "" if num_eval_batches is None else str(num_eval_batches), f"{theta_norm:.10g}", f"{epsilon:.10g}",
        f"{base_loss:.10g}", f"{float(base['acc']):.10g}", f"{kappa_tr:.10g}", f"{kappa_std:.10g}",
        f"{float(kappa_std):.10g}", f"{float(kappa_std):.10g}",
    ]
    if not csv_path.exists():
        csv_path.write_text(header + "\n", encoding="utf-8")
    with open(csv_path, "a", encoding="utf-8") as f:
        f.write(",".join(row) + "\n")
    return json_path
