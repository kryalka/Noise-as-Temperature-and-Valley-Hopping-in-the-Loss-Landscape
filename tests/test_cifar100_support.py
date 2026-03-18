from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import json

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


from ntempvh.data.cifar import get_num_classes_for_dataset
from ntempvh.eval.geometry import compute_geometry
from ntempvh.eval.interpolation import run_interpolation
from ntempvh.train.trainer import train_one_run
from ntempvh.utils.config_validation import validate_train_config



@dataclass
class _DummyLoaders:
    train: DataLoader
    val: DataLoader
    bn: DataLoader



class _TinyNet(nn.Module):
    def __init__(self, num_classes: int = 100):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))



def _make_dummy_loaders(
    *,
    n_train: int = 32,
    n_val: int = 32,
    batch_size: int = 8,
    num_classes: int = 100,
    seed: int = 0,
) -> _DummyLoaders:
    g = torch.Generator().manual_seed(seed)

    x_train = torch.randn((n_train, 3, 32, 32), generator=g)
    y_train = torch.randint(0, num_classes, (n_train,), generator=g)
    x_val = torch.randn((n_val, 3, 32, 32), generator=g)
    y_val = torch.randint(0, num_classes, (n_val,), generator=g)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    bn_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return _DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)



def _make_easy_loaders(
    *,
    n_train: int = 24,
    n_val: int = 24,
    batch_size: int = 8,
    num_classes: int = 100,
) -> _DummyLoaders:
    y_train = torch.arange(n_train, dtype=torch.long) % num_classes
    y_val = torch.arange(n_val, dtype=torch.long) % num_classes

    x_train = torch.zeros((n_train, 3, 32, 32), dtype=torch.float32)
    x_val = torch.zeros((n_val, 3, 32, 32), dtype=torch.float32)
    x_train[torch.arange(n_train), 0, 0, y_train] = 1.0
    x_val[torch.arange(n_val), 0, 0, y_val] = 1.0

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    bn_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return _DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)



def _make_easy_model(num_classes: int = 100) -> _TinyNet:
    model = _TinyNet(num_classes=num_classes)
    with torch.no_grad():
        model.fc.weight.zero_()
        model.fc.bias.zero_()
        for cls in range(num_classes):
            model.fc.weight[cls, cls] = 8.0
    return model



def _write_fake_ckpt(
    *,
    root: Path,
    run_name: str,
    epoch: int,
    payload: dict,
) -> Path:
    ckpt_dir = root / "outputs" / "runs_lr_bs_grid_cifar100" / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path



def _write_interp_cfg(path: Path, *, data_root: str) -> None:
    path.write_text(
        dedent(
            f"""\
            data_root: {data_root}

            path:
              type: linear
              num_points: 3
              bn_recalib_batches: 0

            evaluation:
              batch_size: 8
              bn_batch_size: 8
              split: val
              val_size: 16
              split_seed: 7

            data:
              num_workers: 0
              pin_memory: false
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
              num_directions: 2
              eval_batch_size: 8
              num_eval_batches: 1
              bn_recalib_batches: 0

            evaluation:
              val_size: 16
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
        "dataset": "cifar100",
        "model": "resnet18",
        "data_root": "./ignored",
        "data": {
            "val_size": 16,
            "split_seed": 7,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "optimizer": "sgd",
            "epochs": 1,
            "batch_size": 8,
            "learning_rate": 0.1,
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


def test_train_config_and_num_classes_support_cifar100() -> None:
    cfg = _base_train_cfg()
    validate_train_config(cfg)
    assert get_num_classes_for_dataset("cifar10") == 10
    assert get_num_classes_for_dataset("cifar100") == 100



def test_train_one_run_supports_cifar100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.train.trainer as trainer_mod

    fixed_loaders = _make_dummy_loaders(n_train=16, n_val=16, batch_size=8, num_classes=100, seed=123)
    seen: dict[str, object] = {"called": False, "num_classes": []}

    def fake_get_cifar100_loaders(*, root, batch_size, num_workers=2, pin_memory=True):
        del root, batch_size, num_workers, pin_memory
        seen["called"] = True
        return fixed_loaders

    def fake_get_cifar100_test_loader(*, root, batch_size=256, num_workers=0, pin_memory=True):
        del root, batch_size, num_workers, pin_memory
        return fixed_loaders.val

    def fake_make_model(name: str, num_classes: int):
        seen["num_classes"].append(int(num_classes))
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(trainer_mod, "get_cifar100_loaders", fake_get_cifar100_loaders)
    monkeypatch.setattr(trainer_mod, "get_cifar100_test_loader", fake_get_cifar100_test_loader)
    monkeypatch.setattr(trainer_mod, "make_model", fake_make_model)
    monkeypatch.setattr(trainer_mod, "get_device", lambda: torch.device("cpu"))

    run_dir = tmp_path / "run_cifar100"
    final_ckpt = train_one_run(_base_train_cfg(), seed=123, out_dir=str(run_dir))

    assert seen["called"] is True
    assert seen["num_classes"] == [100]
    assert final_ckpt.exists()

    ckpt = torch.load(final_ckpt, map_location="cpu")
    assert ckpt["dataset"] == "cifar100"



def test_run_interpolation_supports_cifar100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.eval.interpolation as interp_mod

    fixed_loaders = _make_dummy_loaders(n_train=16, n_val=16, batch_size=8, num_classes=100, seed=777)
    seen: dict[str, list[int] | bool] = {"called": False, "num_classes": []}

    def fake_get_cifar100_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256, bn_batch_size=None):
        del root, batch_size, num_workers, pin_memory, val_batch_size, bn_batch_size
        seen["called"] = True
        return fixed_loaders

    def fake_make_model(name: str, num_classes: int):
        seen["num_classes"].append(int(num_classes))
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(interp_mod, "get_cifar100_loaders", fake_get_cifar100_loaders)
    monkeypatch.setattr(interp_mod, "make_model", fake_make_model)
    monkeypatch.setattr(interp_mod, "get_device", lambda: torch.device("cpu"))

    run_name = "cifar100_resnet18_seed1__optsgd_lr0.1_bs128_wd0.0005_mom0.9_schnone__dummy"

    torch.manual_seed(1)
    ckpt_a = {
        "model": "resnet18",
        "dataset": "cifar100",
        "seed": 1,
        "epoch": 1,
        "state_dict": _TinyNet(num_classes=100).state_dict(),
    }
    torch.manual_seed(2)
    ckpt_b = {
        "model": "resnet18",
        "dataset": "cifar100",
        "seed": 1,
        "epoch": 2,
        "state_dict": _TinyNet(num_classes=100).state_dict(),
    }

    ckpt_a_path = _write_fake_ckpt(root=tmp_path, run_name=run_name, epoch=1, payload=ckpt_a)
    ckpt_b_path = _write_fake_ckpt(root=tmp_path, run_name=run_name, epoch=2, payload=ckpt_b)

    cfg_path = tmp_path / "interpolation_cifar100.yaml"
    _write_interp_cfg(cfg_path, data_root="./ignored")

    out_csv = run_interpolation(str(ckpt_a_path), str(ckpt_b_path), str(cfg_path), str(tmp_path / "out"))

    assert seen["called"] is True
    assert all(value == 100 for value in seen["num_classes"])
    assert out_csv.exists()

    meta = json.loads(out_csv.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["dataset"] == "cifar100"



def test_compute_geometry_supports_cifar100(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.eval.geometry as geom_mod

    fixed_loaders = _make_easy_loaders(n_train=16, n_val=16, batch_size=8, num_classes=100)
    seen: dict[str, list[int] | bool] = {"called": False, "num_classes": []}

    def fake_get_cifar100_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        del root, batch_size, num_workers, pin_memory, val_batch_size
        seen["called"] = True
        return fixed_loaders

    def fake_make_model(name: str, num_classes: int):
        seen["num_classes"].append(int(num_classes))
        return _make_easy_model(num_classes=num_classes)

    monkeypatch.setattr(geom_mod, "get_cifar100_loaders", fake_get_cifar100_loaders)
    monkeypatch.setattr(geom_mod, "make_model", fake_make_model)
    monkeypatch.setattr(geom_mod, "get_device", lambda: torch.device("cpu"))

    model = _make_easy_model(num_classes=100)
    ckpt = {
        "model": "resnet18",
        "dataset": "cifar100",
        "seed": 0,
        "epoch": 1,
        "state_dict": model.state_dict(),
    }
    ckpt_path = tmp_path / "epoch_001.pt"
    torch.save(ckpt, ckpt_path)

    cfg_path = tmp_path / "geometry_cifar100.yaml"
    _write_geometry_cfg(cfg_path)

    json_path = compute_geometry(str(ckpt_path), str(cfg_path), str(tmp_path / "out"))

    assert seen["called"] is True
    assert all(value == 100 for value in seen["num_classes"])
    assert json_path.exists()

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["dataset"] == "cifar100"
