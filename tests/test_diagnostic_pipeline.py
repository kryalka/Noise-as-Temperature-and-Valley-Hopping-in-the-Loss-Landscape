from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

import ntempvh.pipeline.diagnostic_pipeline as diag_mod
from ntempvh.utils.artifacts import (
    build_geometry_artifact_context,
    build_path_compare_artifact_context,
)
from ntempvh.utils.checkpoints import build_pair_tag
from ntempvh.utils.io import ensure_dir, load_yaml


def _make_ckpt(run_dir: Path, epoch: int) -> Path:
    ckpt_dir = ensure_dir(run_dir / "checkpoints")
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "model": "resnet18",
            "dataset": "cifar10",
            "seed": 0,
            "epoch": int(epoch),
            "state_dict": {},
        },
        ckpt_path,
    )
    return ckpt_path


def _write_pairs_csv(path: Path, pairs: list[tuple[Path, Path]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ckptA", "ckptB"])
        writer.writeheader()
        for ckpt_a, ckpt_b in pairs:
            writer.writerow({"ckptA": str(ckpt_a), "ckptB": str(ckpt_b)})


def _write_diag_cfg(path: Path, *, pairs_csv: Path, out_root: Path, repo_root: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "inputs:",
                f"  pairs_csv: {pairs_csv}",
                "",
                "diagnostics:",
                f"  compare_config: {repo_root / 'configs/eval/path_compare_test.yaml'}",
                f"  geometry_config: {repo_root / 'configs/eval/geometry.yaml'}",
                "  reuse_existing: true",
                "",
                "outputs:",
                f"  out_root: {out_root}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def test_run_diagnostic_pipeline_builds_pair_summary_and_maps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "cifar10_resnet18_seed0__optsgd_lr0.1_bs64_wd0.0005_mom0.9_schcosine__abc123"
    ckpt_050 = _make_ckpt(run_dir, 50)
    ckpt_070 = _make_ckpt(run_dir, 70)
    ckpt_100 = _make_ckpt(run_dir, 100)

    pairs_csv = tmp_path / "pairs.csv"
    _write_pairs_csv(pairs_csv, [(ckpt_050, ckpt_100), (ckpt_070, ckpt_100)])

    out_root = tmp_path / "diag_out"
    cfg_path = tmp_path / "diagnostic.yaml"
    _write_diag_cfg(cfg_path, pairs_csv=pairs_csv, out_root=out_root, repo_root=repo_root)

    compare_calls: list[tuple[str, str]] = []
    geometry_calls: list[str] = []

    def fake_compare_paths(ckpt_a: str, ckpt_b: str, config_path: str, out_dir: str) -> Path:
        compare_calls.append((ckpt_a, ckpt_b))
        cfg = load_yaml(config_path)
        artifact = build_path_compare_artifact_context(ckpt_a, ckpt_b, cfg)
        out_path = ensure_dir(Path(out_dir) / "comparisons") / f"{artifact['stem']}.json"
        epoch_a = int(Path(ckpt_a).stem.split("_")[1])
        epoch_b = int(Path(ckpt_b).stem.split("_")[1])
        payload = {
            "config": artifact["config"],
            "metrics": {
                "chord_DeltaL": float(epoch_a / 100.0),
                "observed_DeltaL": float(epoch_b / 100.0),
                "chord_length": float(epoch_b - epoch_a),
                "observed_length": float(epoch_b - epoch_a + 5),
                "length_ratio": 1.1,
                "length_excess": 5.0,
            },
            "report_metrics": {
                "Peakobs": float(epoch_b / 100.0),
                "Pitchord": 0.1,
                "Pitobs": 0.05,
                "BarrierGap": float((epoch_b - epoch_a) / 100.0),
                "LengthRatio": 1.1,
                "LengthExcess": 5.0,
                "devL1": 0.2,
            },
            "observed_path": {
                "selection": "explicit",
            },
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    def fake_compute_geometry(ckpt_path: str, geometry_cfg_path: str, out_path: str) -> Path:
        geometry_calls.append(str(ckpt_path))
        cfg = load_yaml(geometry_cfg_path)
        artifact = build_geometry_artifact_context(ckpt_path, cfg, failed=False)
        out_json = ensure_dir(out_path) / f"{artifact['stem']}.json"
        epoch = int(Path(ckpt_path).stem.split("_")[1])
        payload = {
            "ckpt": str(ckpt_path),
            "kappa_tr": float(epoch / 10.0),
            "sigma_kappa": float(epoch / 100.0),
            "anisotropy": float(epoch / 100.0),
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_json

    monkeypatch.setattr(diag_mod, "compare_paths", fake_compare_paths)
    monkeypatch.setattr(diag_mod, "compute_geometry", fake_compute_geometry)

    manifest_path = diag_mod.run_diagnostic_pipeline(cfg_path)
    assert manifest_path.exists()
    assert len(compare_calls) == 2
    assert len(geometry_calls) == 3

    pair_rows = list(csv.DictReader(open(out_root / "diagnostic_pairs.csv", encoding="utf-8")))
    regime_rows = list(csv.DictReader(open(out_root / "diagnostic_regime_table.csv", encoding="utf-8")))
    report = json.loads((out_root / "diagnostic_report.json").read_text(encoding="utf-8"))
    regime_maps = json.loads((out_root / "diagnostic_regime_maps.json").read_text(encoding="utf-8"))
    manifest = json.loads((out_root / "diagnostic_manifest.json").read_text(encoding="utf-8"))
    report_md = (out_root / "diagnostic_report.md").read_text(encoding="utf-8")

    assert len(pair_rows) == 2
    assert pair_rows[0]["status"] == "ok"
    assert pair_rows[0]["Peakobs"] != ""
    assert pair_rows[0]["curvature_proxy_mean"] != ""
    assert len(regime_rows) == 1
    assert regime_rows[0]["mean_BarrierGap"] != ""
    assert "mean_curvature_proxy_mean" in regime_maps["metric_maps"]
    assert report["counts"]["num_ok_pairs"] == 2
    assert report["counts"]["num_unique_checkpoints"] == 3
    assert report["tool_scope"]["purpose"] != ""
    assert "pairs csv" in report_md.lower()
    assert "не привязан жёстко" in report_md
    assert manifest["tool_scope"]["reference_presets"] != ""

    second_manifest = diag_mod.run_diagnostic_pipeline(cfg_path)
    assert second_manifest.exists()
    assert len(compare_calls) == 2
    assert len(geometry_calls) == 3


def test_run_diagnostic_pipeline_marks_geometry_partial(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    run_dir = tmp_path / "cifar10_resnet18_seed0__optsgd_lr0.1_bs64_wd0.0005_mom0.9_schcosine__abc123"
    ckpt_050 = _make_ckpt(run_dir, 50)
    ckpt_100 = _make_ckpt(run_dir, 100)

    pairs_csv = tmp_path / "pairs.csv"
    _write_pairs_csv(pairs_csv, [(ckpt_050, ckpt_100)])

    out_root = tmp_path / "diag_out"
    cfg_path = tmp_path / "diagnostic.yaml"
    _write_diag_cfg(cfg_path, pairs_csv=pairs_csv, out_root=out_root, repo_root=repo_root)

    def fake_compare_paths(ckpt_a: str, ckpt_b: str, config_path: str, out_dir: str) -> Path:
        cfg = load_yaml(config_path)
        artifact = build_path_compare_artifact_context(ckpt_a, ckpt_b, cfg)
        out_json = ensure_dir(Path(out_dir) / "comparisons") / f"{artifact['stem']}.json"
        payload = {
            "config": artifact["config"],
            "metrics": {
                "chord_DeltaL": 0.1,
                "observed_DeltaL": 0.2,
                "chord_length": 10.0,
                "observed_length": 12.0,
                "length_ratio": 1.2,
                "length_excess": 2.0,
            },
            "report_metrics": {
                "Peakobs": 0.2,
                "Pitchord": 0.05,
                "Pitobs": 0.03,
                "BarrierGap": 0.1,
                "LengthRatio": 1.2,
                "LengthExcess": 2.0,
                "devL1": 0.4,
            },
            "observed_path": {"selection": "all"},
        }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_json

    def fake_compute_geometry(ckpt_path: str, geometry_cfg_path: str, out_path: str) -> Path:
        cfg = load_yaml(geometry_cfg_path)
        failed = int(Path(ckpt_path).stem.split("_")[1]) == 100
        artifact = build_geometry_artifact_context(ckpt_path, cfg, failed=failed)
        out_json = ensure_dir(out_path) / f"{artifact['stem']}.json"
        if failed:
            payload = {
                "status": "failed",
                "reason": "synthetic_failure",
                "ckpt": str(ckpt_path),
            }
        else:
            payload = {
                "ckpt": str(ckpt_path),
                "kappa_tr": 5.0,
                "sigma_kappa": 0.5,
                "anisotropy": 0.5,
            }
        out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_json

    monkeypatch.setattr(diag_mod, "compare_paths", fake_compare_paths)
    monkeypatch.setattr(diag_mod, "compute_geometry", fake_compute_geometry)

    diag_mod.run_diagnostic_pipeline(cfg_path)

    pair_rows = list(csv.DictReader(open(out_root / "diagnostic_pairs.csv", encoding="utf-8")))
    report = json.loads((out_root / "diagnostic_report.json").read_text(encoding="utf-8"))

    assert len(pair_rows) == 1
    assert pair_rows[0]["status"] == "geometry_partial"
    assert report["counts"]["num_geometry_partial_pairs"] == 1
    assert report["issues"]
