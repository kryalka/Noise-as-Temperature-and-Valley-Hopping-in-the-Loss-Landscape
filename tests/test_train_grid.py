from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ntempvh.pipeline.train_grid import build_train_grid_jobs
from ntempvh.utils.config_validation import validate_train_grid_config


def _base_train_cfg() -> dict:
    return {
        "dataset": "cifar10",
        "model": "resnet18",
        "data_root": "./ignored",
        "data": {
            "val_size": 5000,
            "split_seed": 42,
            "num_workers": 0,
            "pin_memory": False,
        },
        "training": {
            "optimizer": "sgd",
            "epochs": 20,
            "batch_size": 8,
            "learning_rate": 0.2,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "nesterov": True,
            "scheduler": "none",
        },
        "intervention": {
            "enabled": True,
            "start_epoch": 4,
            "end_epoch": 6,
            "lr_multiplier": 2.0,
            "batch_size": 4,
        },
        "logging": {
            "save_every_epochs": 1,
            "save_final": True,
            "save_best": True,
        },
    }



def test_validate_train_grid_config_rejects_duplicate_variant_names() -> None:
    with pytest.raises(ValueError, match="Duplicate intervention variant name"):
        validate_train_grid_config(
            {
                "base_config": "configs/train/windowed_intervention.yaml",
                "seeds": [1],
                "learning_rates": [0.1],
                "batch_sizes": [64],
                "intervention_variants": [
                    {"name": "same", "start_epoch": 2, "end_epoch": 3, "lr_multiplier": 2.0},
                    {"name": "same", "start_epoch": 4, "end_epoch": 5, "lr_multiplier": 1.5},
                ],
            }
        )



def test_validate_train_grid_config_accepts_arbitrary_intervention_windows() -> None:
    validate_train_grid_config(
        {
            "base_config": "configs/train/windowed_intervention.yaml",
            "seeds": [1],
            "learning_rates": [0.1],
            "batch_sizes": [64],
            "config_overrides": {
                "training": {
                    "epochs": 90,
                }
            },
            "intervention_windows": [
                "12:19",
                {"name": "late_long", "start_epoch": 41, "end_epoch": 58},
                {"name": "short_hot", "start_epoch": 73, "end_epoch": 75, "batch_size": 32},
            ],
            "intervention_lr_multipliers": [0.85, 1.4],
        }
    )



def test_build_train_grid_jobs_supports_config_overrides_and_intervention_variants(
    tmp_path: Path,
) -> None:
    project_root = tmp_path
    base_cfg_path = project_root / "base_train.yaml"
    base_cfg_path.write_text(json.dumps(_base_train_cfg(), ensure_ascii=False, indent=2), encoding="utf-8")

    grid_cfg_path = project_root / "custom_grid.yaml"
    grid_cfg_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_cfg_path.relative_to(project_root)),
                "out_root": "outputs/runs_custom_grid",
                "seeds": [11, 13],
                "learning_rates": [0.03],
                "batch_sizes": [48],
                "config_overrides": {
                    "dataset": "cifar100",
                    "model": "resnet34",
                    "training": {
                        "epochs": 24,
                    },
                },
                "intervention_variants": [
                    {
                        "name": "early_hot",
                        "enabled": True,
                        "start_epoch": 5,
                        "end_epoch": 8,
                        "lr_multiplier": 1.5,
                        "batch_size": 32,
                    },
                    {
                        "name": "late_hot",
                        "enabled": True,
                        "start_epoch": 10,
                        "end_epoch": 12,
                        "lr_multiplier": 2.0,
                        "batch_size": 24,
                    },
                ],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    jobs = build_train_grid_jobs(
        grid_cfg_path,
        tmp_cfg_dir=project_root / ".tmp",
        project_root=project_root,
    )

    assert len(jobs) == 4
    cfg_paths = sorted({job.cfg_path for job in jobs})
    assert len(cfg_paths) == 2
    assert any("early_hot" in cfg_path for cfg_path in cfg_paths)
    assert any("late_hot" in cfg_path for cfg_path in cfg_paths)

    early_cfg = yaml.safe_load((project_root / cfg_paths[0]).read_text(encoding="utf-8"))
    late_cfg = yaml.safe_load((project_root / cfg_paths[1]).read_text(encoding="utf-8"))
    cfgs = [early_cfg, late_cfg]

    for cfg in cfgs:
        assert cfg["dataset"] == "cifar100"
        assert cfg["model"] == "resnet34"
        assert cfg["training"]["learning_rate"] == 0.03
        assert cfg["training"]["batch_size"] == 48
        assert cfg["training"]["epochs"] == 24

    intervention_windows = sorted(
        [
            (
                cfg["intervention"]["start_epoch"],
                cfg["intervention"]["end_epoch"],
                cfg["intervention"]["lr_multiplier"],
                cfg["intervention"]["batch_size"],
            )
            for cfg in cfgs
        ]
    )
    assert intervention_windows == [
        (5, 8, 1.5, 32),
        (10, 12, 2.0, 24),
    ]



def test_build_train_grid_jobs_supports_arbitrary_window_sweeps(tmp_path: Path) -> None:
    project_root = tmp_path
    base_cfg_path = project_root / "base_train.yaml"
    base_cfg_path.write_text(json.dumps(_base_train_cfg(), ensure_ascii=False, indent=2), encoding="utf-8")

    grid_cfg_path = project_root / "window_grid.yaml"
    grid_cfg_path.write_text(
        yaml.safe_dump(
            {
                "base_config": str(base_cfg_path.relative_to(project_root)),
                "out_root": "outputs/runs_custom_window_grid",
                "seeds": [11],
                "learning_rates": [0.03],
                "batch_sizes": [48],
                "config_overrides": {
                    "training": {
                        "epochs": 90,
                    },
                },
                "intervention_windows": [
                    "12:19",
                    {"name": "late_long", "start_epoch": 41, "end_epoch": 58},
                    {"name": "short_hot", "start_epoch": 73, "end_epoch": 75, "batch_size": 24},
                ],
                "intervention_lr_multipliers": [0.85, 1.4],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    jobs = build_train_grid_jobs(
        grid_cfg_path,
        tmp_cfg_dir=project_root / ".tmp",
        project_root=project_root,
    )

    assert len(jobs) == 6
    cfg_paths = sorted({job.cfg_path for job in jobs})
    assert len(cfg_paths) == 6
    assert any("w12_19_x0p85" in cfg_path for cfg_path in cfg_paths)
    assert any("late_long_x1p4" in cfg_path for cfg_path in cfg_paths)
    assert any("short_hot_x0p85" in cfg_path for cfg_path in cfg_paths)

    cfgs = [
        yaml.safe_load((project_root / cfg_path).read_text(encoding="utf-8"))
        for cfg_path in cfg_paths
    ]
    windows = sorted(
        (
            cfg["intervention"]["start_epoch"],
            cfg["intervention"]["end_epoch"],
            cfg["intervention"]["lr_multiplier"],
            cfg["intervention"]["batch_size"],
        )
        for cfg in cfgs
    )
    assert windows == [
        (12, 19, 0.85, 4),
        (12, 19, 1.4, 4),
        (41, 58, 0.85, 4),
        (41, 58, 1.4, 4),
        (73, 75, 0.85, 24),
        (73, 75, 1.4, 24),
    ]



def test_repo_custom_example_grid_configs_expand(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    baseline_jobs = build_train_grid_jobs(
        project_root / "configs/train/lr_bs_grid_custom_example.yaml",
        tmp_cfg_dir=tmp_path / "baseline_tmp",
        project_root=project_root,
    )
    intervention_jobs = build_train_grid_jobs(
        project_root / "configs/train/intervention_lr_bs_grid_custom_example.yaml",
        tmp_cfg_dir=tmp_path / "intervention_tmp",
        project_root=project_root,
    )
    intervention_window_jobs = build_train_grid_jobs(
        project_root / "configs/train/intervention_lr_bs_grid_windows_example.yaml",
        tmp_cfg_dir=tmp_path / "intervention_window_tmp",
        project_root=project_root,
    )

    assert len(baseline_jobs) == 8
    assert len(intervention_jobs) == 8
    assert len(intervention_window_jobs) == 6
    assert any("early_boost" in job.cfg_path for job in intervention_jobs)
    assert any("late_boost" in job.cfg_path for job in intervention_jobs)
    assert any("w12_19_x0p85" in job.cfg_path for job in intervention_window_jobs)
    assert any("late_long_x1p4" in job.cfg_path for job in intervention_window_jobs)



def test_report_and_resnet34_grid_presets_validate(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[1]

    resnet34_jobs = build_train_grid_jobs(
        project_root / "configs/train/lr_bs_grid_resnet34.yaml",
        tmp_cfg_dir=tmp_path / "resnet34_tmp",
        project_root=project_root,
    )
    resnet34_cifar100_jobs = build_train_grid_jobs(
        project_root / "configs/train/lr_bs_grid_resnet34_cifar100.yaml",
        tmp_cfg_dir=tmp_path / "resnet34_cifar100_tmp",
        project_root=project_root,
    )
    report_intervention_jobs = build_train_grid_jobs(
        project_root / "configs/train/intervention_lr_bs_grid_report.yaml",
        tmp_cfg_dir=tmp_path / "report_intervention_tmp",
        project_root=project_root,
    )
    report_intervention_cifar100_jobs = build_train_grid_jobs(
        project_root / "configs/train/intervention_lr_bs_grid_report_cifar100.yaml",
        tmp_cfg_dir=tmp_path / "report_intervention_cifar100_tmp",
        project_root=project_root,
    )

    assert len(resnet34_jobs) == 32
    assert len(resnet34_cifar100_jobs) == 32
    assert len(report_intervention_jobs) == 1536
    assert len(report_intervention_cifar100_jobs) == 1536
    sample_resnet34_cfg = yaml.safe_load((project_root / resnet34_jobs[0].cfg_path).read_text(encoding="utf-8"))
    sample_resnet34_cifar100_cfg = yaml.safe_load(
        (project_root / resnet34_cifar100_jobs[0].cfg_path).read_text(encoding="utf-8")
    )
    assert sample_resnet34_cfg["model"] == "resnet34"
    assert sample_resnet34_cifar100_cfg["model"] == "resnet34"
    assert sample_resnet34_cifar100_cfg["dataset"] == "cifar100"
    assert any("w50_100_x0p7" in job.cfg_path for job in report_intervention_jobs)
    assert any("w80_85_x1p75" in job.cfg_path for job in report_intervention_jobs)
    assert any("w50_100_x0p7" in job.cfg_path for job in report_intervention_cifar100_jobs)
    assert any("w80_85_x1p75" in job.cfg_path for job in report_intervention_cifar100_jobs)
