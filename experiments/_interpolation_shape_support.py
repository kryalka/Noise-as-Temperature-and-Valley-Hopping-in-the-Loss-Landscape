from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ntempvh.utils.checkpoints import parse_checkpoint_path

BARRIER_EPS = 0.01
VALLEY_EPS = 0.01

SUMMARY_COLUMNS = [
    "interp_csv",
    "meta_json",
    "run_name",
    "seed",
    "learning_rate",
    "batch_size",
    "optimizer",
    "weight_decay",
    "momentum",
    "scheduler",
    "ckptA",
    "ckptB",
    "epoch_A",
    "epoch_B",
    "num_points",
    "bn_recalib_batches",
    "L0",
    "L1",
    "A0",
    "A1",
    "max_L",
    "max_t",
    "min_L",
    "min_t",
    "baseline_max_diff",
    "baseline_max_diff_t",
    "baseline_min_diff",
    "baseline_min_diff_t",
    "valley_depth",
    "endpoints_gap_abs",
    "endpoints_gap_rel",
    "endpoint_best_loss",
    "endpoint_worst_loss",
    "middle_best_improvement_over_best_endpoint",
    "middle_best_improvement_over_worst_endpoint",
    "acc_max",
    "acc_max_t",
    "acc_min",
    "acc_min_t",
    "shape_class",
    "monotonicity_class",
    "num_sign_changes_loss",
]


def classify_shape(barrier_height: float, valley_depth: float) -> str:
    barrier_ok = barrier_height >= BARRIER_EPS
    valley_ok = valley_depth >= VALLEY_EPS

    if barrier_ok and not valley_ok:
        return "hump"
    if valley_ok and not barrier_ok:
        return "valley"
    if barrier_ok and valley_ok:
        return "mixed"
    return "flat"


def count_sign_changes(vals: np.ndarray, eps: float = 1e-12) -> int:
    delta = np.diff(vals)
    signs = []
    for value in delta:
        if value > eps:
            signs.append(1)
        elif value < -eps:
            signs.append(-1)

    if not signs:
        return 0

    changes = 0
    prev = signs[0]
    for sign in signs[1:]:
        if sign != prev:
            changes += 1
            prev = sign
    return changes


def monotonic_label(vals: np.ndarray, eps: float = 1e-12) -> str:
    delta = np.diff(vals)
    all_noninc = np.all(delta <= eps)
    all_nondec = np.all(delta >= -eps)

    if all_noninc:
        return "monotone_decreasing"
    if all_nondec:
        return "monotone_increasing"

    sign_changes = count_sign_changes(vals, eps=eps)
    if sign_changes == 1:
        first = next((value for value in delta if abs(value) > eps), 0.0)
        last = next((value for value in delta[::-1] if abs(value) > eps), 0.0)
        if first < 0 and last > 0:
            return "u_shape"
        if first > 0 and last < 0:
            return "hill_shape"

    return "complex"


def build_shape_row(
    *,
    csv_path: Path,
    meta_path: Path,
    meta: dict[str, Any],
    df: pd.DataFrame,
) -> dict[str, Any]:
    t = df["t"].to_numpy(dtype=float)
    loss = df["val_loss"].to_numpy(dtype=float)
    acc = df["val_acc"].to_numpy(dtype=float)

    loss_a = float(loss[0])
    loss_b = float(loss[-1])
    acc_a = float(acc[0])
    acc_b = float(acc[-1])

    baseline = (1.0 - t) * loss_a + t * loss_b
    diff = loss - baseline

    max_idx = int(np.argmax(loss))
    min_idx = int(np.argmin(loss))
    barrier_idx = int(np.argmax(diff))
    valley_idx = int(np.argmin(diff))

    max_loss = float(loss[max_idx])
    min_loss = float(loss[min_idx])
    barrier_height = float(max(0.0, diff[barrier_idx]))
    valley_depth = float(max(0.0, -diff[valley_idx]))

    info_a = parse_checkpoint_path(meta["ckptA"])
    info_b = parse_checkpoint_path(meta["ckptB"])

    return {
        "interp_csv": str(csv_path),
        "meta_json": str(meta_path),
        "run_name": info_a["run_name"],
        "seed": info_a["seed"],
        "learning_rate": info_a["learning_rate"],
        "batch_size": info_a["batch_size"],
        "optimizer": info_a["optimizer"],
        "weight_decay": info_a["weight_decay"],
        "momentum": info_a["momentum"],
        "scheduler": info_a["scheduler"],
        "ckptA": meta["ckptA"],
        "ckptB": meta["ckptB"],
        "epoch_A": int(meta.get("epoch_A", info_a["epoch"])),
        "epoch_B": int(meta.get("epoch_B", info_b["epoch"])),
        "num_points": int((meta.get("path") or {}).get("num_points", len(df))),
        "bn_recalib_batches": int((meta.get("path") or {}).get("bn_recalib_batches", 0)),
        "L0": loss_a,
        "L1": loss_b,
        "A0": acc_a,
        "A1": acc_b,
        "max_L": max_loss,
        "max_t": float(t[max_idx]),
        "min_L": min_loss,
        "min_t": float(t[min_idx]),
        "baseline_max_diff": barrier_height,
        "baseline_max_diff_t": float(t[barrier_idx]),
        "baseline_min_diff": -valley_depth,
        "baseline_min_diff_t": float(t[valley_idx]),
        "valley_depth": valley_depth,
        "endpoints_gap_abs": abs(loss_b - loss_a),
        "endpoints_gap_rel": abs(loss_b - loss_a) / max(abs(loss_a), abs(loss_b), 1e-12),
        "endpoint_best_loss": min(loss_a, loss_b),
        "endpoint_worst_loss": max(loss_a, loss_b),
        "middle_best_improvement_over_best_endpoint": max(0.0, min(loss_a, loss_b) - min_loss),
        "middle_best_improvement_over_worst_endpoint": max(0.0, max(loss_a, loss_b) - min_loss),
        "acc_max": float(np.max(acc)),
        "acc_max_t": float(t[int(np.argmax(acc))]),
        "acc_min": float(np.min(acc)),
        "acc_min_t": float(t[int(np.argmin(acc))]),
        "shape_class": classify_shape(barrier_height, valley_depth),
        "monotonicity_class": monotonic_label(loss),
        "num_sign_changes_loss": count_sign_changes(loss),
    }

def build_shape_summary(
    *,
    interp_root: Path,
    out_csv: Path,
    out_df: pd.DataFrame,
    bad_files: list[dict[str, str]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "interp_root": str(interp_root),
        "out_csv": str(out_csv),
        "num_pairs_total": int(len(out_df)),
        "num_bad_files": int(len(bad_files)),
        "bad_file_examples": bad_files[:20],
        "barrier_eps": BARRIER_EPS,
        "valley_eps": VALLEY_EPS,
    }

    if not len(out_df):
        return summary

    summary["shape_class_counts"] = out_df["shape_class"].value_counts().sort_index().to_dict()
    summary["monotonicity_class_counts"] = (
        out_df["monotonicity_class"].value_counts().sort_index().to_dict()
    )

    by_epoch = (
        out_df.groupby(["epoch_A", "epoch_B"])["shape_class"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary["by_epoch_pair_shape_counts"] = by_epoch.to_dict(orient="records")

    by_lr = (
        out_df.groupby("learning_rate")["shape_class"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    summary["by_learning_rate_shape_counts"] = by_lr.to_dict(orient="records")
    return summary


def print_shape_summary(*, out_csv: Path, out_json: Path, out_df: pd.DataFrame, bad_files: list[dict[str, str]]) -> None:
    print(f"saved interpolation-shapes csv: {out_csv}")
    print(f"saved interpolation-shapes json: {out_json}")
    print(f"pairs total: {len(out_df)}")
    print(f"bad files: {len(bad_files)}")

    if len(out_df):
        print("\nshape counts:")
        print(out_df["shape_class"].value_counts())
        print("\nmonotonicity counts:")
        print(out_df["monotonicity_class"].value_counts())
