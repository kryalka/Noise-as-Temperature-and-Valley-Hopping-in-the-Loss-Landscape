from __future__ import annotations

import argparse
import copy
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset as hf_load_dataset
from torchvision import models
from tqdm.auto import tqdm


SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)

DEFAULT_WINDOWS = [(5, 10), (10, 15), (15, 20), (20, 30), (30, 40), (40, 50)]
CHECKPOINT_EPOCH_PATTERNS = [
    re.compile(r"checkpoint_epoch_(\d+)\.pt$"),
    re.compile(r"epoch_(\d+)\.pt$"),
]
RUN_NAME_RE = re.compile(
    r"^svhn_resnet18_seed(?P<seed>\d+)_lr(?P<lr>[0-9.]+)_bs(?P<bs>\d+)$"
)


@dataclass(frozen=True)
class AnalysisConfig:
    windows: list[tuple[int, int]]
    num_points: int
    bn_recalib_batches: int
    eval_split: str
    eval_batch_size: int
    bn_batch_size: int
    val_size: int
    split_seed: int
    num_workers: int
    pin_memory: bool
    top_k_windows_to_plot: int


class BatchTensorLoader:
    def __init__(
        self,
        *,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: torch.Tensor,
        batch_size: int,
    ):
        self.images = images
        self.labels = labels
        self.indices = indices.to(dtype=torch.long, device="cpu")
        self.batch_size = int(batch_size)
        mean = torch.tensor(SVHN_MEAN, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(SVHN_STD, dtype=torch.float32).view(3, 1, 1)
        self.mean = mean
        self.std = std

    def __len__(self) -> int:
        return max((len(self.indices) + self.batch_size - 1) // self.batch_size, 0)

    def __iter__(self):
        for start in range(0, len(self.indices), self.batch_size):
            batch_idx = self.indices[start : start + self.batch_size]
            images = self.images[batch_idx].float().div(255.0)
            images = (images - self.mean) / self.std
            targets = self.labels[batch_idx]
            yield images, targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix and recompute local SVHN chord-vs-observed analysis."
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("svhn"),
        help="Directory with local SVHN run folders.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Directory for local SVHN dataset download/cache.",
    )
    parser.add_argument(
        "--runs",
        nargs="*",
        default=[],
        help=(
            "Specific run directory names to recompute. "
            "If omitted, all run directories under --runs-root are recomputed."
        ),
    )
    parser.add_argument(
        "--skip-recompute",
        action="store_true",
        help="Only normalize metadata and rebuild aggregate tables without recomputing profiles.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu"],
        help="Computation device. Local script currently supports CPU only.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Optional worker count recorded in analysis metadata.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=21,
        help="Number of interpolation points per path.",
    )
    parser.add_argument(
        "--bn-recalib-batches",
        type=int,
        default=20,
        help="Number of batches for BN recalibration at each path point.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=256,
        help="Evaluation batch size for path profiles.",
    )
    parser.add_argument(
        "--bn-batch-size",
        type=int,
        default=256,
        help="BN recalibration batch size.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=5000,
        help="Validation split size used in analysis.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="Train/val split seed used in analysis.",
    )
    parser.add_argument(
        "--top-k-windows-to-plot",
        type=int,
        default=3,
        help="How many windows to show in the per-run summary plot.",
    )
    parser.add_argument(
        "--analysis-subdir",
        default="chord_vs_observed",
        help="Analysis subdirectory name under each run's analysis/ folder.",
    )
    parser.add_argument(
        "--aggregate-dirname",
        default="aggregate",
        help="Directory name created under --runs-root for cross-run summary tables.",
    )
    return parser.parse_args()


def safe_torch_load(path: Path | str, map_location: str = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model


def analysis_seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def parse_epoch_from_checkpoint_path(path: Path | str) -> int:
    path = Path(path)
    for pattern in CHECKPOINT_EPOCH_PATTERNS:
        match = pattern.search(path.name)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"Could not parse epoch from checkpoint path: {path}")


def list_epoch_checkpoints(checkpoints_dir: Path) -> dict[int, Path]:
    epoch_to_path: dict[int, Path] = {}
    for path in sorted(checkpoints_dir.glob("*.pt")):
        if path.name == "latest_checkpoint.pt":
            continue
        try:
            epoch = parse_epoch_from_checkpoint_path(path)
        except ValueError:
            continue
        epoch_to_path[epoch] = path
    if not epoch_to_path:
        raise FileNotFoundError(f"No epoch checkpoints found in {checkpoints_dir}")
    return dict(sorted(epoch_to_path.items()))


def clone_state_dict_to_cpu(state_dict: dict[str, Any]) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value):
            cloned[key] = value.detach().cpu().clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def extract_state_dict_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    if "state_dict" in checkpoint:
        return clone_state_dict_to_cpu(checkpoint["state_dict"])
    if "model" in checkpoint and hasattr(checkpoint["model"], "state_dict"):
        return clone_state_dict_to_cpu(checkpoint["model"].state_dict())
    raise ValueError("Checkpoint does not contain a usable state_dict or model object")


def load_checkpoint_record(path: Path) -> dict[str, Any]:
    checkpoint = safe_torch_load(path, map_location="cpu")
    return {
        "path": Path(path),
        "epoch": int(checkpoint.get("epoch", parse_epoch_from_checkpoint_path(path))),
        "state_dict": extract_state_dict_from_checkpoint(checkpoint),
    }


def state_dict_params_to_vector(state_dict: dict[str, torch.Tensor]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            continue
        is_bn_buffer = (
            "running_mean" in key or "running_var" in key or "num_batches_tracked" in key
        )
        if is_bn_buffer or not value.is_floating_point():
            continue
        pieces.append(value.detach().reshape(-1).cpu())
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat(pieces)


def state_dict_l2_distance(
    state_dict_a: dict[str, torch.Tensor],
    state_dict_b: dict[str, torch.Tensor],
) -> float:
    vec_a = state_dict_params_to_vector(state_dict_a)
    vec_b = state_dict_params_to_vector(state_dict_b)
    if vec_a.shape != vec_b.shape:
        raise ValueError(f"State dict shape mismatch: {vec_a.shape} vs {vec_b.shape}")
    return float(torch.norm(vec_b - vec_a, p=2).item())


def lerp_state_dict(
    state_dict_a: dict[str, Any],
    state_dict_b: dict[str, Any],
    t_value: float,
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in state_dict_a.keys():
        a_value = state_dict_a[key]
        b_value = state_dict_b[key]
        is_bn_buffer = (
            "running_mean" in key or "running_var" in key or "num_batches_tracked" in key
        )
        if is_bn_buffer or (
            hasattr(a_value, "is_floating_point") and not a_value.is_floating_point()
        ):
            out[key] = a_value.clone()
            continue
        out[key] = (1.0 - t_value) * a_value + t_value * b_value
    return out


def interp_state_dicts_by_breakpoints(
    state_dicts: list[dict[str, Any]],
    breakpoints: np.ndarray,
    t_value: float,
) -> dict[str, Any]:
    clipped_t = float(np.clip(t_value, 0.0, 1.0))
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
