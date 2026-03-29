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


@torch.no_grad()
def recalibrate_bn_stats(
    model: nn.Module,
    loader: BatchTensorLoader | None,
    device: torch.device,
    num_batches: int,
) -> None:
    bn_layers = [
        module for module in model.modules() if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if not bn_layers or loader is None or num_batches <= 0:
        return

    was_training = model.training
    saved_momenta: dict[nn.Module, float | None] = {}
    model.eval()

    for layer in bn_layers:
        saved_momenta[layer] = layer.momentum
        layer.reset_running_stats()
        layer.momentum = None
        layer.train()

    batches_seen = 0
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _ = model(images)
        batches_seen += 1
        if batches_seen >= num_batches:
            break

    for layer in bn_layers:
        layer.momentum = saved_momenta[layer]
    model.train(was_training)


@torch.inference_mode()
def evaluate_loss_acc_loader(
    model: nn.Module,
    loader: BatchTensorLoader,
    device: torch.device,
) -> tuple[float, float]:
    criterion_sum = nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    loss_sum = 0.0
    correct = 0
    total_examples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss_sum += float(criterion_sum(logits, targets).item())
        correct += int((logits.argmax(dim=1) == targets).sum().item())
        total_examples += int(images.size(0))

    return loss_sum / total_examples, correct / total_examples


def build_analysis_loaders(
    *,
    data_root: Path,
    cfg: AnalysisConfig,
) -> tuple[BatchTensorLoader, BatchTensorLoader]:
    train_images, train_labels = load_or_prepare_tensor_split(data_root=data_root, split="train")
    test_images, test_labels = load_or_prepare_tensor_split(data_root=data_root, split="test")
    num_samples = int(train_labels.shape[0])
    if cfg.val_size < 0 or cfg.val_size >= num_samples:
        raise ValueError(f"val_size must be in [0, {num_samples - 1}], got {cfg.val_size}")

    split_generator = torch.Generator()
    split_generator.manual_seed(int(cfg.split_seed))
    permutation = torch.randperm(num_samples, generator=split_generator)
    val_idx = permutation[: cfg.val_size].clone()
    train_idx = permutation[cfg.val_size :].clone()

    bn_loader = BatchTensorLoader(
        images=train_images,
        labels=train_labels,
        indices=train_idx,
        batch_size=int(cfg.bn_batch_size),
    )

    if str(cfg.eval_split).lower() == "val":
        eval_loader = BatchTensorLoader(
            images=train_images,
            labels=train_labels,
            indices=val_idx,
            batch_size=int(cfg.eval_batch_size),
        )
    elif str(cfg.eval_split).lower() == "test":
        eval_loader = BatchTensorLoader(
            images=test_images,
            labels=test_labels,
            indices=torch.arange(test_labels.shape[0], dtype=torch.long),
            batch_size=int(cfg.eval_batch_size),
        )
    else:
        raise ValueError("eval_split must be either 'val' or 'test'")

    return bn_loader, eval_loader


def load_or_prepare_tensor_split(
    *,
    data_root: Path,
    split: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_dir = data_root / "tensor_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    images_path = cache_dir / f"{split}_images_uint8.pt"
    labels_path = cache_dir / f"{split}_labels.pt"

    if images_path.exists() and labels_path.exists():
        images = torch.load(images_path, map_location="cpu")
        labels = torch.load(labels_path, map_location="cpu")
        return images, labels

    hf_cache_dir = data_root / "hf_cache"
    hf_dataset = hf_load_dataset(
        "ufldl-stanford/svhn",
        "cropped_digits",
        split=split,
        cache_dir=str(hf_cache_dir),
    )

    images = torch.empty((len(hf_dataset), 3, 32, 32), dtype=torch.uint8)
    labels = torch.empty((len(hf_dataset),), dtype=torch.long)

    for idx in tqdm(range(len(hf_dataset)), desc=f"materialize {split}", leave=False):
        example = hf_dataset[int(idx)]
        arr = np.asarray(example["image"], dtype=np.uint8)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image for {split}[{idx}], got shape {arr.shape}")
        images[idx] = torch.from_numpy(np.moveaxis(arr, -1, 0))
        labels[idx] = int(example["label"])

    torch.save(images, images_path)
    torch.save(labels, labels_path)
    return images, labels


def build_observed_breakpoints(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, list[float], float, float]:
    state_dicts = [record["state_dict"] for record in records]
    segment_lengths: list[float] = []

    for idx in range(len(state_dicts) - 1):
        segment_lengths.append(
            state_dict_l2_distance(state_dicts[idx], state_dicts[idx + 1])
        )

    observed_length = float(sum(segment_lengths))
    if observed_length > 1e-12:
        breakpoints = [0.0]
        acc = 0.0
        for segment_length in segment_lengths:
            acc += segment_length
            breakpoints.append(acc / observed_length)
    else:
        breakpoints = np.linspace(0.0, 1.0, len(state_dicts)).tolist()

    chord_length = state_dict_l2_distance(state_dicts[0], state_dicts[-1])
    return np.asarray(breakpoints, dtype=np.float64), segment_lengths, chord_length, observed_length


def evaluate_profile(
    *,
    path_type: str,
    records: list[dict[str, Any]],
    breakpoints: np.ndarray | None,
    bn_loader: BatchTensorLoader,
    eval_loader: BatchTensorLoader,
    cfg: AnalysisConfig,
    device: torch.device,
    window_label: str,
) -> pd.DataFrame:
    model = build_model().to(device)
    state_dicts = [record["state_dict"] for record in records]
    rows: list[dict[str, float]] = []

    progress = tqdm(
        np.linspace(0.0, 1.0, int(cfg.num_points)),
        desc=f"{window_label} | {path_type}",
        leave=False,
    )

    for t_value in progress:
        t_float = float(t_value)
        if path_type == "linear":
            interpolated_state = lerp_state_dict(state_dicts[0], state_dicts[-1], t_float)
        elif path_type == "observed":
            if breakpoints is None:
                raise ValueError("Observed path requires breakpoints")
            interpolated_state = interp_state_dicts_by_breakpoints(
                state_dicts,
                breakpoints,
                t_float,
            )
        else:
            raise ValueError(f"Unsupported path type: {path_type}")

        model.load_state_dict(interpolated_state, strict=True)
        recalibrate_bn_stats(
            model,
            bn_loader,
            device,
            num_batches=int(cfg.bn_recalib_batches),
        )
        loss_value, acc_value = evaluate_loss_acc_loader(model, eval_loader, device)
        rows.append(
            {
                "t": t_float,
                "loss": float(loss_value),
                "acc": float(acc_value),
            }
        )
        progress.set_postfix(loss=f"{loss_value:.4f}", acc=f"{acc_value:.4f}")

    return pd.DataFrame(rows)


def compute_profile_shape_metrics(profile_df: pd.DataFrame) -> dict[str, float]:
    t_values = profile_df["t"].to_numpy(dtype=np.float64)
    loss_values = profile_df["loss"].to_numpy(dtype=np.float64)
    baseline = (1.0 - t_values) * float(loss_values[0]) + t_values * float(loss_values[-1])
    diff = loss_values - baseline
    peak_idx = int(np.argmax(diff))
    pit_idx = int(np.argmin(diff))
    pit_signed = float(diff[pit_idx])
    return {
        "peak": float(max(0.0, diff[peak_idx])),
        "peak_t": float(t_values[peak_idx]),
        "pit_signed": pit_signed,
        "pit_depth": float(max(0.0, -pit_signed)),
        "pit_t": float(t_values[pit_idx]),
    }


def compute_profile_deviation_metrics(
    chord_df: pd.DataFrame,
    observed_df: pd.DataFrame,
) -> dict[str, float]:
    chord_t = chord_df["t"].to_numpy(dtype=np.float64)
    chord_loss = chord_df["loss"].to_numpy(dtype=np.float64)
    observed_t = observed_df["t"].to_numpy(dtype=np.float64)
    observed_loss = observed_df["loss"].to_numpy(dtype=np.float64)

    shared_points = max(len(chord_t), len(observed_t), 2)
    shared_t = np.linspace(0.0, 1.0, shared_points)
    chord_interp = np.interp(shared_t, chord_t, chord_loss)
    observed_interp = np.interp(shared_t, observed_t, observed_loss)
    abs_diff = np.abs(observed_interp - chord_interp)

    return {
        "devL1": float(np.mean(abs_diff)),
        "devLinf": float(np.max(abs_diff)),
    }


def analyze_window(
    *,
    window: tuple[int, int],
    epoch_to_path: dict[int, Path],
    bn_loader: BatchTensorLoader,
    eval_loader: BatchTensorLoader,
    profiles_dir: Path,
    cfg: AnalysisConfig,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    start_epoch, end_epoch = int(window[0]), int(window[1])
    if end_epoch <= start_epoch:
        raise ValueError(f"Window must satisfy end > start, got {window}")

    required_epochs = list(range(start_epoch, end_epoch + 1))
    missing_epochs = [epoch for epoch in required_epochs if epoch not in epoch_to_path]
    if missing_epochs:
        raise FileNotFoundError(f"Missing checkpoints for window {window}: {missing_epochs}")

    observed_records = [load_checkpoint_record(epoch_to_path[epoch]) for epoch in required_epochs]
    chord_records = [observed_records[0], observed_records[-1]]
    breakpoints, segment_lengths, chord_length, observed_length = build_observed_breakpoints(
        observed_records
    )

    window_label = f"{start_epoch}->{end_epoch}"
    chord_df = evaluate_profile(
        path_type="linear",
        records=chord_records,
        breakpoints=None,
        bn_loader=bn_loader,
        eval_loader=eval_loader,
        cfg=cfg,
        device=device,
        window_label=window_label,
    )
    observed_df = evaluate_profile(
        path_type="observed",
        records=observed_records,
        breakpoints=breakpoints,
        bn_loader=bn_loader,
        eval_loader=eval_loader,
        cfg=cfg,
        device=device,
        window_label=window_label,
    )

    chord_shape = compute_profile_shape_metrics(chord_df)
    observed_shape = compute_profile_shape_metrics(observed_df)
    deviation_metrics = compute_profile_deviation_metrics(chord_df, observed_df)
    length_ratio = float(observed_length / max(chord_length, 1e-12))

    chord_path = profiles_dir / f"window_{start_epoch:03d}_{end_epoch:03d}__chord.csv"
    observed_path = profiles_dir / f"window_{start_epoch:03d}_{end_epoch:03d}__observed.csv"
    chord_df.to_csv(chord_path, index=False)
    observed_df.to_csv(observed_path, index=False)

    result = {
        "window": window_label,
        "epoch_start": start_epoch,
        "epoch_end": end_epoch,
        "Peak_chord": float(chord_shape["peak"]),
        "Peak_obs": float(observed_shape["peak"]),
        "BarrierGap": float(observed_shape["peak"] - chord_shape["peak"]),
        "Pit_chord": float(chord_shape["pit_signed"]),
        "Pit_chord_depth": float(chord_shape["pit_depth"]),
        "Pit_obs": float(observed_shape["pit_signed"]),
        "Pit_obs_depth": float(observed_shape["pit_depth"]),
        "devL1": float(deviation_metrics["devL1"]),
        "devLinf": float(deviation_metrics["devLinf"]),
        "L_chord": float(chord_length),
        "L_obs": float(observed_length),
        "LengthRatio": float(length_ratio),
        "LengthExcess": float(observed_length - chord_length),
        "chord_peak_t": float(chord_shape["peak_t"]),
        "observed_peak_t": float(observed_shape["peak_t"]),
        "chord_pit_t": float(chord_shape["pit_t"]),
        "observed_pit_t": float(observed_shape["pit_t"]),
        "segment_lengths": [float(value) for value in segment_lengths],
        "chord_profile_csv": str(chord_path.resolve()),
        "observed_profile_csv": str(observed_path.resolve()),
    }
    return result, chord_df, observed_df


def select_windows_for_plot(summary_df: pd.DataFrame, top_k: int) -> list[str]:
    candidates = summary_df[
        (summary_df["BarrierGap"] > 0.0) & (summary_df["Pit_chord"] < 0.0)
    ].copy()
    if candidates.empty:
        candidates = summary_df.copy()
    ranked = candidates.sort_values(["BarrierGap", "Pit_chord"], ascending=[False, True])
    return ranked["window"].head(int(top_k)).tolist()


def plot_selected_windows(
    profile_store: dict[str, dict[str, pd.DataFrame]],
    selected_windows: list[str],
    plot_path: Path,
    eval_split: str,
) -> None:
    if not selected_windows:
        return

    fig, axes = plt.subplots(1, len(selected_windows), figsize=(6 * len(selected_windows), 4))
    if len(selected_windows) == 1:
        axes = [axes]

    for ax, window_label in zip(axes, selected_windows):
        chord_df = profile_store[window_label]["chord"]
        observed_df = profile_store[window_label]["observed"]
        ax.plot(chord_df["t"], chord_df["loss"], label="chord", linewidth=2)
        ax.plot(observed_df["t"], observed_df["loss"], label="observed", linewidth=2)
        ax.set_title(window_label)
        ax.set_xlabel("t")
        ax.set_ylabel(f"{eval_split}_loss")
        ax.grid(alpha=0.3)

    axes[0].legend()
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def parse_run_name(run_name: str) -> dict[str, Any]:
    match = RUN_NAME_RE.match(run_name)
    if match is None:
        raise ValueError(f"Unsupported run name format: {run_name}")
    return {
        "seed": int(match.group("seed")),
        "lr": float(match.group("lr")),
        "bs": int(match.group("bs")),
    }


def normalize_summary_metadata(run_dir: Path) -> None:
    run_name = run_dir.name
    history_path = run_dir / "history.csv"
    summary_path = run_dir / "summary.json"
    latest_checkpoint = (run_dir / "latest_checkpoint.pt").resolve()

    last_epoch = None
    best_test_accuracy = None
    if history_path.exists():
        history = pd.read_csv(history_path)
        if not history.empty:
            last_epoch = int(history["epoch"].iloc[-1])
            best_test_accuracy = float(history["test_accuracy"].max())

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    if last_epoch is None:
        ckpts = list_epoch_checkpoints(run_dir / "checkpoints")
        last_epoch = max(ckpts)

    summary.update(
        {
            "experiment_name": run_name,
            "last_epoch": int(last_epoch),
            "best_test_accuracy": (
                float(best_test_accuracy)
                if best_test_accuracy is not None
                else summary.get("best_test_accuracy")
            ),
            "latest_checkpoint": str(latest_checkpoint),
            "run_dir": str(run_dir.resolve()),
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_analysis_config(
    *,
    analysis_dir: Path,
    run_dir: Path,
    cfg: AnalysisConfig,
    analysis_subdir: str,
) -> None:
    payload = {
        "analysis_run_dir": str(run_dir.resolve()),
        "analysis_subdir": str(analysis_subdir),
        "windows": [list(window) for window in cfg.windows],
        "num_points": int(cfg.num_points),
        "bn_recalib_batches": int(cfg.bn_recalib_batches),
        "eval_split": str(cfg.eval_split),
        "eval_batch_size": int(cfg.eval_batch_size),
        "bn_batch_size": int(cfg.bn_batch_size),
        "val_size": int(cfg.val_size),
        "split_seed": int(cfg.split_seed),
        "num_workers": int(cfg.num_workers),
        "pin_memory": bool(cfg.pin_memory),
        "top_k_windows_to_plot": int(cfg.top_k_windows_to_plot),
        "save_outputs": True,
    }
    (analysis_dir / "analysis_config.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def recompute_run_analysis(
    *,
    run_dir: Path,
    data_root: Path,
    cfg: AnalysisConfig,
    device: torch.device,
    analysis_subdir: str,
) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    analysis_dir = run_dir / "analysis" / str(analysis_subdir)
    profiles_dir = analysis_dir / "profiles"
    plots_dir = analysis_dir / "plots"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    write_analysis_config(
        analysis_dir=analysis_dir,
        run_dir=run_dir,
        cfg=cfg,
        analysis_subdir=analysis_subdir,
    )
    normalize_summary_metadata(run_dir)

    epoch_to_path = list_epoch_checkpoints(checkpoints_dir)
    bn_loader, eval_loader = build_analysis_loaders(data_root=data_root, cfg=cfg)

    analysis_rows: list[dict[str, Any]] = []
    analysis_profiles: dict[str, dict[str, pd.DataFrame]] = {}

    for window in cfg.windows:
        result, chord_df, observed_df = analyze_window(
            window=window,
            epoch_to_path=epoch_to_path,
            bn_loader=bn_loader,
            eval_loader=eval_loader,
            profiles_dir=profiles_dir,
            cfg=cfg,
            device=device,
        )
        analysis_rows.append(result)
        analysis_profiles[result["window"]] = {
            "chord": chord_df,
            "observed": observed_df,
        }

    analysis_summary_df = pd.DataFrame(analysis_rows)
    column_order = [
        "window",
        "Peak_chord",
        "Peak_obs",
        "BarrierGap",
        "Pit_chord",
        "Pit_chord_depth",
        "Pit_obs",
        "Pit_obs_depth",
        "devL1",
        "devLinf",
        "L_chord",
        "L_obs",
        "LengthRatio",
        "LengthExcess",
        "chord_profile_csv",
        "observed_profile_csv",
    ]
    analysis_summary_df = analysis_summary_df[column_order]
    summary_path = analysis_dir / "svhn_chord_vs_observed_summary.csv"
    analysis_summary_df.to_csv(summary_path, index=False)

    selected_plot_windows = select_windows_for_plot(
        analysis_summary_df,
        top_k=cfg.top_k_windows_to_plot,
    )
    plot_selected_windows(
        analysis_profiles,
        selected_plot_windows,
        plots_dir / "loss_profiles_top_windows.png",
        eval_split=cfg.eval_split,
    )

    print(
        "Saved:",
        summary_path,
        f"| windows={len(analysis_summary_df)}",
        f"| positive BarrierGap={(analysis_summary_df['BarrierGap'] > 0).sum()}",
        f"| negative Pit_chord={(analysis_summary_df['Pit_chord'] < 0).sum()}",
    )


def normalize_only(run_dir: Path, cfg: AnalysisConfig, analysis_subdir: str) -> None:
    analysis_dir = run_dir / "analysis" / str(analysis_subdir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    write_analysis_config(
        analysis_dir=analysis_dir,
        run_dir=run_dir,
        cfg=cfg,
        analysis_subdir=analysis_subdir,
    )
    normalize_summary_metadata(run_dir)


def aggregate_runs(
    runs_root: Path,
    *,
    analysis_subdir: str,
    aggregate_dirname: str,
) -> tuple[Path, Path, Path]:
    rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        run_name = run_dir.name
        try:
            parsed = parse_run_name(run_name)
        except ValueError:
            continue

        summary_path = (
            run_dir / "analysis" / str(analysis_subdir) / "svhn_chord_vs_observed_summary.csv"
        )
        if not summary_path.exists():
            continue

        df = pd.read_csv(summary_path)
        rows.append(
            {
                "run": run_name,
                "lr": float(parsed["lr"]),
                "bs": int(parsed["bs"]),
                "T_proxy_lr_over_bs": float(parsed["lr"] / parsed["bs"]),
                "mean Peak_chord": float(df["Peak_chord"].mean()),
                "mean Peak_obs": float(df["Peak_obs"].mean()),
                "mean BarrierGap": float(df["BarrierGap"].mean()),
                "mean Pit_chord": float(df["Pit_chord"].mean()),
                "mean devL1": float(df["devL1"].mean()),
                "num_windows": int(len(df)),
                "n BarrierGap>0": int((df["BarrierGap"] > 0).sum()),
                "n Pit_chord<0": int((df["Pit_chord"] < 0).sum()),
            }
        )

        for _, row in df.iterrows():
            window_rows.append(
                {
                    "run": run_name,
                    "lr": float(parsed["lr"]),
                    "bs": int(parsed["bs"]),
                    "T_proxy_lr_over_bs": float(parsed["lr"] / parsed["bs"]),
                    "window": row["window"],
                    "Peak_chord": float(row["Peak_chord"]),
                    "Peak_obs": float(row["Peak_obs"]),
                    "BarrierGap": float(row["BarrierGap"]),
                    "Pit_chord": float(row["Pit_chord"]),
                    "devL1": float(row["devL1"]),
                }
            )

    aggregate_dir = runs_root / str(aggregate_dirname)
    aggregate_dir.mkdir(parents=True, exist_ok=True)

    run_summary_df = pd.DataFrame(rows).sort_values(["lr", "bs"]).reset_index(drop=True)
    window_metrics_df = pd.DataFrame(window_rows).sort_values(["window", "bs"]).reset_index(drop=True)
    barriergap_by_bs_df = (
        window_metrics_df.pivot_table(index="window", columns="bs", values="BarrierGap", aggfunc="mean")
        .reset_index()
    )

    run_summary_path = aggregate_dir / "svhn_run_summary.csv"
    window_metrics_path = aggregate_dir / "svhn_window_metrics.csv"
    barriergap_by_bs_path = aggregate_dir / "svhn_window_barriergap_by_bs.csv"

    run_summary_df.to_csv(run_summary_path, index=False)
    window_metrics_df.to_csv(window_metrics_path, index=False)
    barriergap_by_bs_df.to_csv(barriergap_by_bs_path, index=False)

    return run_summary_path, window_metrics_path, barriergap_by_bs_path
