from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import yaml

from ntempvh.data.cifar import (
    get_num_classes_for_dataset,
    get_supported_image_datasets,
    get_svhn_loaders,
    get_svhn_test_loader,
)
from ntempvh.models.resnet_cifar import get_supported_model_names, make_model
from ntempvh.pipeline.train_grid import build_train_grid_jobs
from ntempvh.train.trainer import train_one_run
from ntempvh.utils.config_validation import validate_train_config



class _TinyNet(nn.Module):
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x):
        return self.fc(self.flatten(x))



def _make_loaders(*, train_batch_size: int, val_batch_size: int, num_classes: int = 10):
    x_train = torch.zeros((24, 3, 32, 32), dtype=torch.float32)
    y_train = torch.arange(24, dtype=torch.long) % num_classes
    x_val = torch.zeros((12, 3, 32, 32), dtype=torch.float32)
    y_val = torch.arange(12, dtype=torch.long) % num_classes

    x_train[torch.arange(24), 0, 0, y_train] = 1.0
    x_val[torch.arange(12), 0, 0, y_val] = 1.0

    train_ds = torch.utils.data.TensorDataset(x_train, y_train)
    val_ds = torch.utils.data.TensorDataset(x_val, y_val)

    return type(
        "DummyLoaders",
        (),
        {
            "train": torch.utils.data.DataLoader(train_ds, batch_size=int(train_batch_size), shuffle=False, num_workers=0),
            "val": torch.utils.data.DataLoader(val_ds, batch_size=int(val_batch_size), shuffle=False, num_workers=0),
            "bn": torch.utils.data.DataLoader(train_ds, batch_size=int(val_batch_size), shuffle=False, num_workers=0),
        },
    )()



def _base_train_cfg() -> dict:
    return {
        "dataset": "svhn",
        "model": "resnet50",
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
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
            "scheduler": "none",
        },
        "logging": {
            "save_every_epochs": 1,
            "save_final": True,
            "save_best": True,
        },
    }


def test_supported_dataset_and_model_names_include_new_choices() -> None:
    assert "svhn" in get_supported_image_datasets()
    assert "resnet50" in get_supported_model_names()
    assert "resnet101" in get_supported_model_names()
    assert get_num_classes_for_dataset("svhn") == 10


def test_make_model_supports_resnet50_and_keeps_cifar_stem() -> None:
    model = make_model("resnet50", num_classes=11)
    assert model.fc.out_features == 11
    assert model.conv1.kernel_size == (3, 3)
    assert model.conv1.stride == (1, 1)


def test_validate_train_config_accepts_svhn_and_resnet50() -> None:
    cfg = _base_train_cfg()
    validate_train_config(cfg)



def test_svhn_and_resnet50_example_configs_validate(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    svhn_cfg = yaml.safe_load((project_root / "configs/train/sgd_svhn.yaml").read_text(encoding="utf-8"))
    resnet50_cfg = yaml.safe_load((project_root / "configs/train/sgd_resnet50.yaml").read_text(encoding="utf-8"))
    svhn_resnet50_cfg = yaml.safe_load(
        (project_root / "configs/train/sgd_svhn_resnet50.yaml").read_text(encoding="utf-8")
    )
    validate_train_config(svhn_cfg)
    validate_train_config(resnet50_cfg)
    validate_train_config(svhn_resnet50_cfg)

    svhn_jobs = build_train_grid_jobs(
        project_root / "configs/train/lr_bs_grid_svhn.yaml",
        tmp_cfg_dir=tmp_path / "svhn_grid_tmp",
        project_root=project_root,
    )
    assert len(svhn_jobs) == 32
    sample_cfg = yaml.safe_load((project_root / svhn_jobs[0].cfg_path).read_text(encoding="utf-8"))
    assert sample_cfg["dataset"] == "svhn"



def test_svhn_wrappers_use_generic_dataset_entrypoints(monkeypatch: pytest.MonkeyPatch) -> None:
    import ntempvh.data.cifar as cifar_mod

    seen: list[tuple[str, str]] = []

    def fake_get_dataset_loaders(dataset_name, root, batch_size, **kwargs):
        del root, kwargs
        seen.append(("train", dataset_name))
        return _make_loaders(train_batch_size=int(batch_size), val_batch_size=8)

    def fake_get_dataset_test_loader(dataset_name, root, *, batch_size=256, **kwargs):
        del root, kwargs
        seen.append(("test", dataset_name))
        return _make_loaders(train_batch_size=8, val_batch_size=int(batch_size)).val

    monkeypatch.setattr(cifar_mod, "get_dataset_loaders", fake_get_dataset_loaders)
    monkeypatch.setattr(cifar_mod, "get_dataset_test_loader", fake_get_dataset_test_loader)

    loaders = get_svhn_loaders("./ignored", batch_size=16)
    test_loader = get_svhn_test_loader("./ignored", batch_size=32)

    assert loaders.train.batch_size == 16
    assert test_loader.batch_size == 32
    assert seen == [("train", "svhn"), ("test", "svhn")]



def test_trainer_supports_generic_dataset_and_model_fallbacks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.train.trainer as trainer_mod

    seen: dict[str, list[str]] = {"train": [], "test": []}

    def fake_get_dataset_loaders(dataset_name, root, batch_size, **kwargs):
        del root, kwargs
        seen["train"].append(str(dataset_name))
        return _make_loaders(train_batch_size=int(batch_size), val_batch_size=8)

    def fake_get_dataset_test_loader(dataset_name, root, *, batch_size=256, **kwargs):
        del root, kwargs
        seen["test"].append(str(dataset_name))
        return _make_loaders(train_batch_size=8, val_batch_size=int(batch_size)).val

    monkeypatch.setattr(trainer_mod, "get_dataset_loaders", fake_get_dataset_loaders)
    monkeypatch.setattr(trainer_mod, "get_dataset_test_loader", fake_get_dataset_test_loader)
    monkeypatch.setattr(trainer_mod, "make_model", lambda name, num_classes: _TinyNet(num_classes))
    monkeypatch.setattr(trainer_mod, "get_device", lambda: torch.device("cpu"))

    run_dir = tmp_path / "svhn_run"
    final_ckpt = train_one_run(_base_train_cfg(), seed=11, out_dir=str(run_dir))

    assert final_ckpt.exists()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["final_test_acc"] is not None
    assert summary["train_test_gap"] is not None
    assert seen["train"] == ["svhn"]
    assert seen["test"] == ["svhn"]



def test_interpolation_and_geometry_support_generic_dataset_fallbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    import ntempvh.eval.geometry as geom_mod
    import ntempvh.eval.interpolation as interp_mod

    seen: list[tuple[str, str]] = []

    def fake_get_dataset_loaders(dataset_name, root, batch_size, **kwargs):
        del root, batch_size, kwargs
        seen.append(("train", str(dataset_name)))
        return "train_loader"

    def fake_get_dataset_test_loader(dataset_name, root, *, batch_size=256, **kwargs):
        del root, batch_size, kwargs
        seen.append(("test", str(dataset_name)))
        return "test_loader"

    monkeypatch.setattr(interp_mod, "get_dataset_loaders", fake_get_dataset_loaders)
    monkeypatch.setattr(interp_mod, "get_dataset_test_loader", fake_get_dataset_test_loader)
    monkeypatch.setattr(geom_mod, "get_dataset_loaders", fake_get_dataset_loaders)

    assert interp_mod._select_train_loader_fn("svhn")(root="ignored", batch_size=8) == "train_loader"
    assert interp_mod._select_test_loader_fn("svhn")(root="ignored", batch_size=8) == "test_loader"
    assert geom_mod._select_train_loader_fn("svhn")(root="ignored", batch_size=8) == "train_loader"
    assert seen == [("train", "svhn"), ("test", "svhn"), ("train", "svhn")]
