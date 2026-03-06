from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

import pytest

from ntempvh.eval.intervention_geometry import run_intervention_geometry_batch
from ntempvh.utils.artifacts import build_geometry_artifact_context
from ntempvh.utils.intervention_runs import resolve_intervention_geometry_checkpoints
from ntempvh.utils.io import load_yaml

def _run_name(*, seed: int, lr: float, bs: int) -> str:
    return (
        f"cifar10_resnet18_seed{seed}"
        f"__optsgd_lr{lr:g}_bs{bs}_wd0_mom0_schnone__deadbeef"
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



def _write_run_artifacts(
    *,
    root: Path,
    run_name: str,
    epochs_total: int,
    intervention_enabled: bool,
    intervention_start_epoch: int | None = None,
    intervention_end_epoch: int | None = None,
    intervention_lr_multiplier: float | None = None,
    intervention_batch_size: int | None = None,
    epoch_checkpoints: list[int] | tuple[int, ...] = (),
) -> Path:
    run_dir = root / run_name
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epoch_checkpoints:
        (ckpt_dir / f"epoch_{int(epoch):03d}.pt").write_text("stub", encoding="utf-8")

    final_ckpt = ckpt_dir / "final.pt"
    final_ckpt.write_text("stub-final", encoding="utf-8")

    run_config = {
        "seed": 1,
        "dataset": "cifar10",
        "model": "resnet18",
        "training": {
            "optimizer": "sgd",
            "epochs": int(epochs_total),
            "batch_size": 8,
            "learning_rate": 0.2,
            "momentum": 0.0,
            "weight_decay": 0.0,
            "nesterov": False,
            "scheduler": "none",
        },
    }
    if intervention_enabled:
        run_config["intervention"] = {
            "enabled": True,
            "start_epoch": int(intervention_start_epoch),
            "end_epoch": int(intervention_end_epoch),
            "lr_multiplier": float(intervention_lr_multiplier),
            "batch_size": (
                None if intervention_batch_size is None else int(intervention_batch_size)
            ),
        }

    summary = {
        "seed": 1,
        "epochs": int(epochs_total),
        "final_checkpoint": str(final_ckpt),
        "final_val_loss": 0.5,
        "final_val_acc": 0.8,
        "intervention": {
            "enabled": bool(intervention_enabled),
            "start_epoch": intervention_start_epoch,
            "end_epoch": intervention_end_epoch,
            "lr_multiplier": intervention_lr_multiplier,
            "batch_size": intervention_batch_size,
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

    return run_dir



def test_resolve_intervention_geometry_checkpoints_selects_pre_post_final(
    tmp_path: Path,
) -> None:
    run_dir = _write_run_artifacts(
        root=tmp_path,
        run_name=_run_name(seed=1, lr=0.2, bs=8),
        epochs_total=4,
        intervention_enabled=True,
        intervention_start_epoch=2,
        intervention_end_epoch=3,
        intervention_lr_multiplier=2.0,
        intervention_batch_size=4,
        epoch_checkpoints=[1, 2, 3, 4],
    )

    selection = resolve_intervention_geometry_checkpoints(run_dir)

    assert selection.run_name == run_dir.name
    assert selection.intervention_start_epoch == 2
    assert selection.intervention_end_epoch == 3
    assert selection.intervention_lr_multiplier == 2.0
    assert selection.intervention_batch_size == 4
    assert [ckpt.checkpoint_role for ckpt in selection.checkpoints] == ["pre", "post", "final"]
    assert [ckpt.checkpoint_epoch for ckpt in selection.checkpoints] == [1, 3, 4]
    assert selection.checkpoints[0].checkpoint_path.endswith("epoch_001.pt")
    assert selection.checkpoints[1].checkpoint_path.endswith("epoch_003.pt")
    assert selection.checkpoints[2].checkpoint_path.endswith("final.pt")



def test_resolve_intervention_geometry_checkpoints_fails_when_required_epoch_missing(
    tmp_path: Path,
) -> None:
    run_dir = _write_run_artifacts(
        root=tmp_path,
        run_name=_run_name(seed=1, lr=0.2, bs=8),
        epochs_total=4,
        intervention_enabled=True,
        intervention_start_epoch=2,
        intervention_end_epoch=3,
        intervention_lr_multiplier=2.0,
        intervention_batch_size=4,
        epoch_checkpoints=[2, 3, 4],
    )

    with pytest.raises(FileNotFoundError, match="theta_pre"):
        resolve_intervention_geometry_checkpoints(run_dir)



def test_run_intervention_geometry_batch_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntempvh.eval.intervention_geometry as batch_mod

    runs_root = tmp_path / "runs"
    intervention_run = _write_run_artifacts(
        root=runs_root,
        run_name=_run_name(seed=1, lr=0.2, bs=8),
        epochs_total=4,
        intervention_enabled=True,
        intervention_start_epoch=2,
        intervention_end_epoch=3,
        intervention_lr_multiplier=2.0,
        intervention_batch_size=4,
        epoch_checkpoints=[1, 2, 3, 4],
    )
    _write_run_artifacts(
        root=runs_root,
        run_name=_run_name(seed=2, lr=0.2, bs=8),
        epochs_total=4,
        intervention_enabled=False,
        epoch_checkpoints=[1, 2, 3, 4],
    )

    cfg_path = tmp_path / "geometry.yaml"
    _write_geometry_cfg(cfg_path)
    cfg = load_yaml(cfg_path)

    def fake_compute_geometry(ckpt_path: str, geometry_cfg_path: str, out_path: str) -> Path:
        del geometry_cfg_path
        artifact = build_geometry_artifact_context(ckpt_path, cfg, failed=False)
        out_dir = Path(out_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"{artifact['stem']}.json"
        payload = {
            "ckpt": str(ckpt_path),
            "base": {"loss": 0.25, "acc": 0.75},
            "kappa_tr": 1.5,
            "kappa_tr_std": 0.1,
            "sigma_kappa": 0.1,
            "anisotropy": 0.1,
            "artifact": {
                "stem": artifact["stem"],
            },
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return json_path

    monkeypatch.setattr(batch_mod, "compute_geometry", fake_compute_geometry)

    out_dir = tmp_path / "geometry_intervention"
    summary_json = run_intervention_geometry_batch(
        str(runs_root),
        str(cfg_path),
        str(out_dir),
    )

    assert summary_json.exists()
    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["num_runs_scanned"] == 2
    assert summary["num_intervention_runs"] == 1
    assert summary["num_non_intervention_runs"] == 1
    assert summary["num_rows"] == 3
    assert summary["num_success"] == 3
    assert summary["num_failed"] == 0
    assert summary["num_errors"] == 0

    csv_path = out_dir / "intervention_geometry_summary.csv"
    assert csv_path.exists()
    lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 4
    assert intervention_run.name in lines[1]
    assert "pre" in csv_path.read_text(encoding="utf-8")
    assert "post" in csv_path.read_text(encoding="utf-8")
    assert "final" in csv_path.read_text(encoding="utf-8")
    assert "sigma_kappa" in lines[0]
    assert "anisotropy" in lines[0]
