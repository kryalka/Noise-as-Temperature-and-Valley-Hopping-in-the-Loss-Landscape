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

from ntempvh.eval.path_compare import compare_paths
from ntempvh.utils.config_validation import validate_path_compare_config
from ntempvh.utils.io import load_yaml


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



def _make_dummy_loaders(
    *,
    n_train: int = 32,
    n_val: int = 32,
    batch_size: int = 16,
    seed: int = 0,
) -> _DummyLoaders:
    g = torch.Generator().manual_seed(seed)

    x_train = torch.randn((n_train, 3, 32, 32), generator=g)
    y_train = torch.randint(0, 10, (n_train,), generator=g)
    x_val = torch.randn((n_val, 3, 32, 32), generator=g)
    y_val = torch.randint(0, 10, (n_val,), generator=g)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    bn_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return _DummyLoaders(train=train_loader, val=val_loader, bn=bn_loader)



def _write_fake_ckpt(
    *,
    root: Path,
    run_name: str,
    epoch: int,
    payload: dict,
) -> Path:
    ckpt_dir = root / "outputs" / "runs_lr_bs_grid" / run_name / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(payload, ckpt_path)
    return ckpt_path



def _write_compare_cfg(path: Path) -> None:
    path.write_text(
        dedent(
            """\
            data_root: ./ignored

            path:
              num_points: 5
              bn_recalib_batches: 0
              observed:
                selection: all
                milestone_epochs: []
                epochs: []

            barrier:
              definition: max_minus_linear_baseline
              thresholds:
                - 0.01
                - 0.05

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



def test_compare_paths_generates_supported_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.eval.interpolation as interp_mod

    device = torch.device("cpu")
    fixed_loaders = _make_dummy_loaders(n_train=16, n_val=16, batch_size=8, seed=123)

    def fake_get_cifar10_loaders(*, root, batch_size, num_workers=2, pin_memory=True, val_batch_size=256):
        return fixed_loaders

    def fake_make_model(name: str, num_classes: int):
        return _TinyNet(num_classes=num_classes)

    monkeypatch.setattr(interp_mod, "get_cifar10_loaders", fake_get_cifar10_loaders)
    monkeypatch.setattr(interp_mod, "make_model", fake_make_model)
    monkeypatch.setattr(interp_mod, "get_device", lambda: device)

    run_name = "cifar10_resnet18_seed1__optsgd_lr0.1_bs128_wd0.0005_mom0.9_schnone__dummy"

    torch.manual_seed(21)
    ckpt_a = {
        "model": "resnet18",
        "dataset": "cifar10",
        "seed": 1,
        "epoch": 1,
        "state_dict": _TinyNet(num_classes=10).state_dict(),
    }
    torch.manual_seed(22)
    ckpt_mid = {
        "model": "resnet18",
        "dataset": "cifar10",
        "seed": 1,
        "epoch": 2,
        "state_dict": _TinyNet(num_classes=10).state_dict(),
    }
    torch.manual_seed(23)
    ckpt_b = {
        "model": "resnet18",
        "dataset": "cifar10",
        "seed": 1,
        "epoch": 3,
        "state_dict": _TinyNet(num_classes=10).state_dict(),
    }

    ckpt_a_path = _write_fake_ckpt(root=tmp_path, run_name=run_name, epoch=1, payload=ckpt_a)
    _write_fake_ckpt(root=tmp_path, run_name=run_name, epoch=2, payload=ckpt_mid)
    ckpt_b_path = _write_fake_ckpt(root=tmp_path, run_name=run_name, epoch=3, payload=ckpt_b)

    cfg_path = tmp_path / "path_compare.yaml"
    _write_compare_cfg(cfg_path)

    out_dir = tmp_path / "artifacts" / "path_compare"
    json_path = compare_paths(str(ckpt_a_path), str(ckpt_b_path), str(cfg_path), str(out_dir))

    assert json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    chord_interp_csv = Path(payload["artifacts"]["chord_interp_csv"])
    observed_interp_csv = Path(payload["artifacts"]["observed_interp_csv"])
    chord_barrier_json = Path(payload["artifacts"]["chord_barrier_json"])
    observed_barrier_json = Path(payload["artifacts"]["observed_barrier_json"])

    assert chord_interp_csv.exists()
    assert observed_interp_csv.exists()
    assert chord_barrier_json.exists()
    assert observed_barrier_json.exists()

    assert payload["observed_path"]["selection"] == "all"
    assert payload["observed_path"]["resolved_epochs"] == [1, 2, 3]
    assert payload["observed_path"]["parameterization"] == "arc_length_fraction"

    metrics = payload["metrics"]
    for key in [
        "chord_DeltaL",
        "observed_DeltaL",
        "barrier_gap",
        "chord_length",
        "observed_length",
        "length_ratio",
        "length_excess",
        "loss_profile_l1_mean",
        "loss_profile_linf",
    ]:
        assert np.isfinite(float(metrics[key]))

    report_metrics = payload["report_metrics"]
    for key in [
        "Peakobs",
        "Pitchord",
        "Pitobs",
        "BarrierGap",
        "LengthRatio",
        "LengthExcess",
        "devL1",
    ]:
        assert np.isfinite(float(report_metrics[key]))

    assert float(metrics["observed_length"]) >= float(metrics["chord_length"])
    assert float(report_metrics["BarrierGap"]) == float(metrics["barrier_gap"])
    assert float(report_metrics["LengthRatio"]) == float(metrics["length_ratio"])
    assert float(report_metrics["LengthExcess"]) == float(metrics["length_excess"])
    assert float(report_metrics["devL1"]) == float(metrics["loss_profile_l1_mean"])

    summary_csv = out_dir / "comparisons" / "path_comparisons.csv"
    assert summary_csv.exists()
    summary_text = summary_csv.read_text(encoding="utf-8")
    assert str(json_path) in summary_text
    assert str(chord_interp_csv) in summary_text
    assert "Peakobs" in summary_text.splitlines()[0]



def test_validate_path_compare_config_rejects_empty_explicit_observed_epochs() -> None:
    with pytest.raises(ValueError, match="requires a non-empty epochs list"):
        validate_path_compare_config(
            {
                "data_root": "./data",
                "path": {
                    "num_points": 5,
                    "bn_recalib_batches": 0,
                    "observed": {
                        "selection": "explicit",
                        "epochs": [],
                    },
                },
                "barrier": {
                    "definition": "max_minus_linear_baseline",
                    "thresholds": [0.01],
                },
                "evaluation": {
                    "batch_size": 8,
                    "bn_batch_size": 8,
                    "split": "val",
                    "val_size": 16,
                    "split_seed": 7,
                },
                "data": {
                    "num_workers": 0,
                    "pin_memory": False,
                },
            }
        )


def test_repo_custom_example_path_compare_config_validates() -> None:
    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(project_root / "configs/eval/path_compare_custom_example.yaml")
    validate_path_compare_config(cfg)
