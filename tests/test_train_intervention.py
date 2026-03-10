from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import ntempvh.cli as cli_mod
from ntempvh.train.trainer import train_one_run
from ntempvh.utils.config_validation import validate_train_config


@dataclass
class _DummyLoaders:
    train: DataLoader
    val: DataLoader
    bn: DataLoader



class _TinyNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))



def _make_loaders(
    *,
    train_batch_size: int,
    val_batch_size: int = 8,
    n_train: int = 24,
    n_val: int = 12,
    shuffle_seed: int = 0,
) -> _DummyLoaders:
    y_train = torch.arange(n_train, dtype=torch.long) % 10
    y_val = torch.arange(n_val, dtype=torch.long) % 10

    x_train = torch.zeros((n_train, 3, 32, 32), dtype=torch.float32)
    x_val = torch.zeros((n_val, 3, 32, 32), dtype=torch.float32)
    x_train[torch.arange(n_train), 0, 0, y_train] = 1.0
    x_val[torch.arange(n_val), 0, 0, y_val] = 1.0

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    g = torch.Generator().manual_seed(int(shuffle_seed))

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_batch_size),
        shuffle=True,
        generator=g,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=0,
    )
    bn_loader = DataLoader(
        train_ds,
        batch_size=int(val_batch_size),
        shuffle=False,
        num_workers=0,
    )
    return _DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)



def _base_train_cfg() -> dict:
    return {
        "dataset": "cifar10",
        "model": "resnet18",
        "data_root": "./ignored",
        "data": {
            "val_size": 12,
            "split_seed": 7,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "optimizer": "sgd",
            "epochs": 3,
            "batch_size": 8,
            "learning_rate": 0.2,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "scheduler": "none",
        },
        "logging": {
            "save_every_epochs": 0,
            "save_final": True,
            "save_best": False,
        },
    }


@pytest.mark.parametrize(
    ("intervention", "match"),
    [
        (
            {
                "enabled": True,
                "start_epoch": 3,
                "end_epoch": 2,
                "lr_multiplier": 2.0,
                "batch_size": 4,
            },
            "start_epoch must be <= end_epoch",
        ),
        (
            {
                "enabled": True,
                "start_epoch": 2,
                "end_epoch": 5,
                "lr_multiplier": 2.0,
                "batch_size": 4,
            },
            "must be <= train.training.epochs",
        ),
        (
            {
                "enabled": True,
                "start_epoch": 2,
                "end_epoch": 3,
                "lr_multiplier": 0.0,
                "batch_size": 4,
            },
            "lr_multiplier",
        ),
        (
            {
                "enabled": True,
                "start_epoch": 2,
                "end_epoch": 3,
                "lr_multiplier": 2.0,
                "batch_size": 0,
            },
            "batch_size",
        ),
    ],
)

def test_validate_train_config_rejects_invalid_intervention(
    intervention: dict,
    match: str,
) -> None:
    cfg = _base_train_cfg()
    cfg["intervention"] = intervention

    with pytest.raises(ValueError, match=match):
        validate_train_config(cfg)



def test_train_one_run_logs_intervention_effective_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.train.trainer as trainer_mod

    seen_batch_sizes: list[int] = []

    def fake_get_cifar10_loaders(
        *,
        root,
        batch_size,
        val_size=5000,
        split_seed=0,
        shuffle_seed=None,
        num_workers=0,
        pin_memory=True,
        val_batch_size=256,
        bn_batch_size=None,
    ):
        del root, val_size, num_workers, pin_memory, bn_batch_size
        seen_batch_sizes.append(int(batch_size))
        return _make_loaders(
            train_batch_size=int(batch_size),
            val_batch_size=int(val_batch_size),
            shuffle_seed=int(split_seed if shuffle_seed is None else shuffle_seed),
        )

    def fake_get_cifar10_test_loader(
        *,
        root,
        batch_size=256,
        num_workers=0,
        pin_memory=True,
    ):
        del root, num_workers, pin_memory
        return _make_loaders(
            train_batch_size=8,
            val_batch_size=int(batch_size),
            shuffle_seed=123,
        ).val

    monkeypatch.setattr(trainer_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(trainer_mod, "get_cifar10_test_loader", fake_get_cifar10_test_loader)
    monkeypatch.setattr(trainer_mod, "make_model", lambda name, num_classes: _TinyNet(num_classes))
    monkeypatch.setattr(trainer_mod, "get_device", lambda: torch.device("cpu"))

    cfg = _base_train_cfg()
    cfg["intervention"] = {
        "enabled": True,
        "start_epoch": 2,
        "end_epoch": 3,
        "lr_multiplier": 0.5,
        "batch_size": 4,
    }

    run_dir = tmp_path / "run"
    final_ckpt = train_one_run(cfg, seed=123, out_dir=str(run_dir))

    assert final_ckpt.exists()
    assert seen_batch_sizes == [8, 4]

    run_config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert run_config["intervention"]["enabled"] is True
    assert run_config["intervention"]["batch_size"] == 4

    metrics_rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    ]
    assert [row["is_intervention_epoch"] for row in metrics_rows] == [False, True, True]
    assert [row["effective_batch_size"] for row in metrics_rows] == [8, 4, 4]
    assert [row["effective_learning_rate"] for row in metrics_rows] == [0.2, 0.1, 0.1]

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["intervention"]["enabled"] is True
    assert summary["intervention"]["start_epoch"] == 2
    assert summary["intervention"]["end_epoch"] == 3
    assert summary["intervention"]["lr_multiplier"] == 0.5
    assert summary["intervention"]["batch_size"] == 4
    assert summary["intervention"]["effective_batch_size"] == 4
    assert summary["intervention"]["num_intervention_epochs"] == 2
    assert summary["final_train_acc"] is not None
    assert summary["final_test_acc"] is not None
    assert summary["train_test_gap"] == pytest.approx(
        float(summary["final_train_acc"]) - float(summary["final_test_acc"])
    )



def test_format_run_id_distinguishes_enabled_intervention_without_changing_baseline() -> None:
    base_cfg = _base_train_cfg()

    disabled_cfg = {
        **_base_train_cfg(),
        "intervention": {
            "enabled": False,
            "start_epoch": 2,
            "end_epoch": 3,
            "lr_multiplier": 2.0,
            "batch_size": 4,
        },
    }
    enabled_cfg = {
        **_base_train_cfg(),
        "intervention": {
            "enabled": True,
            "start_epoch": 2,
            "end_epoch": 3,
            "lr_multiplier": 2.0,
            "batch_size": 4,
        },
    }

    base_run_id = cli_mod._format_run_id(base_cfg, seed=1)
    disabled_run_id = cli_mod._format_run_id(disabled_cfg, seed=1)
    enabled_run_id = cli_mod._format_run_id(enabled_cfg, seed=1)

    assert disabled_run_id == base_run_id
    assert enabled_run_id != base_run_id
