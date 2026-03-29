from __future__ import annotations

import json
from pathlib import Path


def run_name(*, seed: int, lr: float, bs: int) -> str:
    return (
        f"cifar10_resnet18_seed{seed}"
        f"__optsgd_lr{lr:g}_bs{bs}_wd0_mom0_schnone__deadbeef"
    )



def write_intervention_run(
    *,
    root: Path,
    run_name: str,
    with_metrics: bool = True,
    with_pre_checkpoint: bool = True,
    with_post_checkpoint: bool = True,
) -> Path:
    run_dir = root / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if with_pre_checkpoint:
        (ckpt_dir / "epoch_001.pt").write_text("pre", encoding="utf-8")
    (ckpt_dir / "epoch_002.pt").write_text("mid", encoding="utf-8")
    if with_post_checkpoint:
        (ckpt_dir / "epoch_003.pt").write_text("post", encoding="utf-8")
    (ckpt_dir / "epoch_004.pt").write_text("last-epoch", encoding="utf-8")
    final_ckpt = ckpt_dir / "final.pt"
    final_ckpt.write_text("final", encoding="utf-8")

    run_config = {
        "seed": 1,
        "dataset": "cifar10",
        "model": "resnet18",
        "training": {
            "optimizer": "sgd",
            "epochs": 4,
            "batch_size": 8,
            "learning_rate": 0.2,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "scheduler": "none",
        },
        "intervention": {
            "enabled": True,
            "start_epoch": 2,
            "end_epoch": 3,
            "lr_multiplier": 2.0,
            "batch_size": 4,
        },
    }
    summary = {
        "seed": 1,
        "epochs": 4,
        "final_checkpoint": str(final_ckpt),
        "final_train_loss": 0.6,
        "final_train_acc": 0.9,
        "best_val_loss": 0.35,
        "best_epoch": 3,
        "final_val_loss": 0.4,
        "final_val_acc": 0.85,
        "final_test_loss": 0.38,
        "final_test_acc": 0.83,
        "train_test_gap": 0.07,
        "intervention": {
            "enabled": True,
            "start_epoch": 2,
            "end_epoch": 3,
            "lr_multiplier": 2.0,
            "batch_size": 4,
            "effective_batch_size": 4,
            "num_intervention_epochs": 2,
        },
    }

    (run_dir / "run_config.json").write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if with_metrics:
        metric_rows = [
            {
                "epoch": 1,
                "train_loss": 1.0,
                "train_acc": 0.5,
                "val_loss": 0.9,
                "val_acc": 0.55,
                "lr": 0.2,
                "is_intervention_epoch": False,
                "effective_learning_rate": 0.2,
                "effective_batch_size": 8,
            },
            {
                "epoch": 2,
                "train_loss": 0.8,
                "train_acc": 0.6,
                "val_loss": 0.7,
                "val_acc": 0.65,
                "lr": 0.2,
                "is_intervention_epoch": True,
                "effective_learning_rate": 0.4,
                "effective_batch_size": 4,
            },
            {
                "epoch": 3,
                "train_loss": 0.7,
                "train_acc": 0.7,
                "val_loss": 0.6,
                "val_acc": 0.75,
                "lr": 0.2,
                "is_intervention_epoch": True,
                "effective_learning_rate": 0.4,
                "effective_batch_size": 4,
            },
            {
                "epoch": 4,
                "train_loss": 0.6,
                "train_acc": 0.8,
                "val_loss": 0.4,
                "val_acc": 0.85,
                "lr": 0.2,
                "is_intervention_epoch": False,
                "effective_learning_rate": 0.2,
                "effective_batch_size": 8,
            },
        ]
        (run_dir / "metrics.jsonl").write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in metric_rows) + "\n",
            encoding="utf-8",
        )

    return run_dir
