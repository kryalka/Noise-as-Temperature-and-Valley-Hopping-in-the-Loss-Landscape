from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ntempvh.data.cifar import (
    get_cifar10_loaders,
    get_cifar100_loaders,
    get_cifar10_test_loader,
    get_cifar100_test_loader,
    get_dataset_loaders,
    get_dataset_test_loader,
)
from ntempvh.eval.metrics import state_dict_l2_distance
from ntempvh.utils.checkpoints import resolve_observed_checkpoint_sequence



def select_train_loader_fn(dataset_name: str):
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name == "cifar10":
        return get_cifar10_loaders
    if dataset_name == "cifar100":
        return get_cifar100_loaders

    def loader_fn(*, root, batch_size, val_size=5000, split_seed=0, shuffle_seed=None, num_workers=0, pin_memory=True, val_batch_size=256, bn_batch_size=None):
        return get_dataset_loaders(
            dataset_name,
            root,
            batch_size,
            val_size=val_size,
            split_seed=split_seed,
            shuffle_seed=shuffle_seed,
            num_workers=num_workers,
            pin_memory=pin_memory,
            val_batch_size=val_batch_size,
            bn_batch_size=bn_batch_size,
        )

    return loader_fn



def select_test_loader_fn(dataset_name: str):
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name == "cifar10":
        return get_cifar10_test_loader
    if dataset_name == "cifar100":
        return get_cifar100_test_loader

    def loader_fn(*, root, batch_size=256, num_workers=0, pin_memory=True):
        return get_dataset_test_loader(
            dataset_name,
            root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loader_fn



def validate_sequence_checkpoint(
    seq_ckpt: dict[str, Any],
    *,
    seq_path: str | Path,
    expected_model: str,
    expected_dataset: str,
    reference_state_dict: dict[str, Any],
) -> None:
    seq_model = str(seq_ckpt.get("model", "")).lower()
    if seq_model != expected_model:
        raise ValueError(f"Checkpoint model mismatch for {seq_path}: {seq_model} vs {expected_model}")

    seq_dataset = str(seq_ckpt.get("dataset", "")).lower()
    if seq_dataset != expected_dataset:
        raise ValueError(f"Checkpoint dataset mismatch for {seq_path}: {seq_dataset} vs {expected_dataset}")

    seq_state_dict = seq_ckpt["state_dict"]
    if list(reference_state_dict.keys()) != list(seq_state_dict.keys()):
        raise ValueError(f"Checkpoint state dict keys do not match for {seq_path}")

    for key in reference_state_dict.keys():
        if reference_state_dict[key].shape != seq_state_dict[key].shape:
            raise ValueError(
                f"Checkpoint shape mismatch for key {key} in {seq_path}: "
                f"{seq_state_dict[key].shape} vs {reference_state_dict[key].shape}"
            )



def load_checkpoint_cached(
    cache: dict[str, dict[str, Any]],
    ckpt_path: str | Path,
) -> dict[str, Any]:
    key = str(Path(ckpt_path).resolve())
    if key not in cache:
        cache[key] = torch.load(ckpt_path, map_location="cpu")
    return cache[key]



def build_observed_path_metadata(
    sequence_paths: list[Path],
    sequence_ckpts: list[dict[str, Any]],
) -> tuple[dict[str, Any], np.ndarray]:
    segment_lengths: list[float] = []
    for idx in range(len(sequence_ckpts) - 1):
        seg_len = state_dict_l2_distance(
            sequence_ckpts[idx]["state_dict"],
            sequence_ckpts[idx + 1]["state_dict"],
        )
        segment_lengths.append(float(seg_len))

    total_length = float(sum(segment_lengths))
    if total_length > 1e-12:
        breakpoints = [0.0]
        acc = 0.0
        for seg_len in segment_lengths:
            acc += seg_len
            breakpoints.append(acc / total_length)
    else:
        breakpoints = np.linspace(0.0, 1.0, len(sequence_ckpts)).tolist()

    chord_length = state_dict_l2_distance(
        sequence_ckpts[0]["state_dict"],
        sequence_ckpts[-1]["state_dict"],
    )

    observed_meta = {
        "resolved_checkpoints": [str(path) for path in sequence_paths],
        "resolved_epochs": [int(ckpt["epoch"]) for ckpt in sequence_ckpts],
        "num_checkpoints": int(len(sequence_ckpts)),
        "num_segments": int(max(0, len(sequence_ckpts) - 1)),
        "parameterization": "arc_length_fraction",
        "segment_lengths": [float(value) for value in segment_lengths],
        "segment_endpoints_t": [float(value) for value in breakpoints],
        "total_path_length": float(total_length),
        "chord_length": float(chord_length),
    }
    return observed_meta, np.asarray(breakpoints, dtype=np.float64)



def resolve_path_sequence(
    *,
    ckpt_a: str,
    ckpt_b: str,
    path_cfg: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    checkpoint_a: dict[str, Any],
    checkpoint_b: dict[str, Any],
    expected_model: str,
    expected_dataset: str,
    reference_state_dict: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray | None]:
    path_type = str(path_cfg["type"])
    pivot_paths = list(path_cfg.get("pivots", []))

    path_meta: dict[str, Any] = {
        "type": path_type,
        "num_points": int(path_cfg["num_points"]),
        "bn_recalib_batches": int(path_cfg["bn_recalib_batches"]),
        "pivots": list(pivot_paths),
    }
    if "observed" in path_cfg:
        path_meta["observed"] = dict(path_cfg["observed"])

    if path_type == "observed":
        observed_cfg = dict(path_cfg.get("observed", {}))
        sequence_paths = resolve_observed_checkpoint_sequence(
            ckpt_a,
            ckpt_b,
            selection=str(observed_cfg.get("selection", "all")),
            milestone_epochs=list(observed_cfg.get("milestone_epochs", []) or []),
            epochs=list(observed_cfg.get("epochs", []) or []),
        )

        sequence_ckpts: list[dict[str, Any]] = []
        for seq_path in sequence_paths:
            seq_ckpt = load_checkpoint_cached(cache, seq_path)
            validate_sequence_checkpoint(
                seq_ckpt,
                seq_path=seq_path,
                expected_model=expected_model,
                expected_dataset=expected_dataset,
                reference_state_dict=reference_state_dict,
            )
            sequence_ckpts.append(seq_ckpt)

        observed_meta, breakpoints = build_observed_path_metadata(sequence_paths, sequence_ckpts)
        path_meta["observed"] = {
            **dict(path_meta.get("observed", {})),
            **observed_meta,
        }
        return sequence_ckpts, path_meta, breakpoints

    sequence_ckpts = [checkpoint_a]
    for pivot_path in pivot_paths:
        pivot_ckpt = load_checkpoint_cached(cache, pivot_path)
        validate_sequence_checkpoint(
            pivot_ckpt,
            seq_path=pivot_path,
            expected_model=expected_model,
            expected_dataset=expected_dataset,
            reference_state_dict=reference_state_dict,
        )
        sequence_ckpts.append(pivot_ckpt)
    sequence_ckpts.append(checkpoint_b)
    return sequence_ckpts, path_meta, None
