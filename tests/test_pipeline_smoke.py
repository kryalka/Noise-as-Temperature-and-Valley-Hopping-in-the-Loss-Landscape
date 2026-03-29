from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from textwrap import dedent

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from ntempvh.eval.barrier import compute_barrier
from ntempvh.eval.geometry import compute_geometry
from ntempvh.eval.interpolation import run_interpolation
from ntempvh.train.trainer import train_one_run


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



def _make_separable_loaders(
    *,
    train_batch_size: int,
    val_batch_size: int,
    bn_batch_size: int,
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
        batch_size=int(bn_batch_size),
        shuffle=False,
        num_workers=0,
    )
    return _DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)



def _fake_get_cifar10_loaders(
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
    del root, val_size, num_workers, pin_memory
    return _make_separable_loaders(
        train_batch_size=int(batch_size),
        val_batch_size=int(val_batch_size),
        bn_batch_size=int(bn_batch_size) if bn_batch_size is not None else int(val_batch_size),
        shuffle_seed=int(split_seed if shuffle_seed is None else shuffle_seed),
    )



def _fake_get_cifar10_test_loader(
    *,
    root,
    batch_size=256,
    num_workers=0,
    pin_memory=True,
):
    del root, num_workers, pin_memory
    return _make_separable_loaders(
        train_batch_size=8,
        val_batch_size=int(batch_size),
        bn_batch_size=int(batch_size),
        shuffle_seed=7,
    ).val



def _make_easy_model(num_classes: int = 10) -> _TinyNet:
    model = _TinyNet(num_classes=num_classes)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        for cls in range(num_classes):
            model.fc.weight[cls, cls] = 8.0
    return model


def _run_name(*, seed: int, lr: float, bs: int) -> str:
    return (
        f"cifar10_resnet18_seed{seed}"
        f"__optsgd_lr{lr:g}_bs{bs}_wd0_mom0_schnone__deadbeef"
    )



def _write_interpolation_cfg(path: Path) -> None:
    path.write_text(
        dedent(
            """\
            data_root: ./ignored

            path:
              type: linear
              num_points: 5
              bn_recalib_batches: 0
              pivots: []

            evaluation:
              batch_size: 8
              bn_batch_size: 8
              split: val
              val_size: 12
              split_seed: 7

            data:
              num_workers: 0
              pin_memory: false
            """
        ),
        encoding="utf-8",
    )



def _write_barrier_cfg(path: Path) -> None:
    path.write_text(
        dedent(
            """\
            barrier:
              definition: max_minus_linear_baseline
              thresholds:
                - 0.01
                - 0.05
            """
        ),
        encoding="utf-8",
    )



def _write_geometry_cfg(path: Path) -> None:
    path.write_text(
        dedent(
            """\
            data_root: ./ignored

            geometry:
              alpha: 1e-3
              num_directions: 3
              eval_batch_size: 8
              num_eval_batches: 1
              bn_recalib_batches: 0

            evaluation:
              val_size: 12
              split_seed: 7

            data:
              num_workers: 0
              pin_memory: false
            """
        ),
        encoding="utf-8",
    )



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
            "epochs": 2,
            "batch_size": 8,
            "learning_rate": 0.2,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "scheduler": "none",
        },
        "logging": {
            "save_every_epochs": 1,
            "save_final": True,
            "save_best": False,
        },
    }



def _patch_tiny_pipeline_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    import ntempvh.train.trainer as trainer_mod
    import ntempvh.eval.interpolation as interp_mod
    import ntempvh.eval.geometry as geom_mod

    monkeypatch.setattr(trainer_mod, "get_cifar10_loaders", _fake_get_cifar10_loaders)
    monkeypatch.setattr(trainer_mod, "get_cifar10_test_loader", _fake_get_cifar10_test_loader)
    monkeypatch.setattr(interp_mod, "get_cifar10_loaders", _fake_get_cifar10_loaders)
    monkeypatch.setattr(geom_mod, "get_cifar10_loaders", _fake_get_cifar10_loaders)

    monkeypatch.setattr(trainer_mod, "make_model", lambda name, num_classes: _TinyNet(num_classes))
    monkeypatch.setattr(interp_mod, "make_model", lambda name, num_classes: _TinyNet(num_classes))
    monkeypatch.setattr(geom_mod, "make_model", lambda name, num_classes: _TinyNet(num_classes))

    monkeypatch.setattr(trainer_mod, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(interp_mod, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(geom_mod, "get_device", lambda: torch.device("cpu"))



def test_train_interpolation_barrier_pipeline_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_tiny_pipeline_modules(monkeypatch)

    cfg = _base_train_cfg()
    seed = 11
    run_name = _run_name(seed=seed, lr=cfg["training"]["learning_rate"], bs=cfg["training"]["batch_size"])
    run_dir = tmp_path / "outputs" / "runs_lr_bs_grid" / run_name

    final_ckpt = train_one_run(cfg, seed=seed, out_dir=str(run_dir))

    ckpt_a = run_dir / "checkpoints" / "epoch_001.pt"
    ckpt_b = run_dir / "checkpoints" / "epoch_002.pt"
    summary_path = run_dir / "summary.json"
    run_config_path = run_dir / "run_config.json"
    metrics_path = run_dir / "metrics.jsonl"

    assert ckpt_a.exists()
    assert ckpt_b.exists()
    assert final_ckpt.exists()
    assert summary_path.exists()
    assert run_config_path.exists()
    assert metrics_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["final_checkpoint"] == str(final_ckpt)
    assert summary["save_every_epochs"] == 1

    metrics_lines = metrics_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 2

    interp_cfg = tmp_path / "interpolation.yaml"
    barrier_cfg = tmp_path / "barrier.yaml"
    _write_interpolation_cfg(interp_cfg)
    _write_barrier_cfg(barrier_cfg)

    interp_dir = tmp_path / "artifacts" / "interpolation"
    barrier_dir = tmp_path / "artifacts" / "barrier"

    interp_csv = run_interpolation(str(ckpt_a), str(ckpt_b), str(interp_cfg), str(interp_dir))
    barrier_json = compute_barrier(str(interp_csv), str(barrier_cfg), str(barrier_dir))

    interp_meta_path = interp_csv.with_suffix(".meta.json")
    assert interp_csv.exists()
    assert interp_meta_path.exists()
    assert barrier_json.exists()

    interp_arr = np.loadtxt(interp_csv, delimiter=",", skiprows=1)
    assert interp_arr.shape == (5, 3)

    interp_meta = json.loads(interp_meta_path.read_text(encoding="utf-8"))
    barrier_payload = json.loads(barrier_json.read_text(encoding="utf-8"))

    assert Path(interp_meta["ckptA"]).resolve() == ckpt_a.resolve()
    assert Path(interp_meta["ckptB"]).resolve() == ckpt_b.resolve()
    assert interp_meta["artifact"]["stem"] == interp_csv.stem
    assert interp_meta["artifact"]["pair_tag"] in interp_csv.stem
    assert interp_meta["evaluation"]["bn_batch_size"] == 8
    assert interp_meta["path"]["num_points"] == 5

    assert barrier_payload["interp_csv"] == str(interp_csv)
    assert barrier_payload["artifact"]["interp_stem"] == interp_csv.stem
    assert barrier_payload["artifact"]["stem"] == barrier_json.stem
    assert barrier_payload["artifact"]["pair_tag"] == interp_meta["artifact"]["pair_tag"]
    assert barrier_payload["path"] == interp_meta["path"]
    assert barrier_payload["evaluation"] == interp_meta["evaluation"]

    barriers_csv = barrier_dir / "barriers.csv"
    assert barriers_csv.exists()
    assert str(interp_csv) in barriers_csv.read_text(encoding="utf-8")
    assert "cfg_" in interp_csv.stem
    assert "cfg_" in barrier_json.stem
