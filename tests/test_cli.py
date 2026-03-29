from __future__ import annotations

import json
from pathlib import Path

import pytest

import ntempvh.cli as cli_mod


def test_cli_geometry_reports_failure_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "geometry_failed__stub.json"
    artifact.write_text(
        json.dumps(
            {
                "status": "failed",
                "reason": "unstable_bn_recalibration",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(cli_mod, "compute_geometry", lambda ckpt, config, out: artifact)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ntempvh",
            "geometry",
            "--ckpt",
            "dummy.pt",
            "--config",
            "geometry.yaml",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    cli_mod.main()
    captured = capsys.readouterr()

    assert "saved geometry failure artifact:" in captured.out
    assert "geometry failure reason: unstable_bn_recalibration" in captured.out
    assert "saved geometry:" not in captured.out



def test_cli_compare_paths_reports_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "pathcompare__stub.json"
    artifact.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_mod, "compare_paths", lambda ckpt_a, ckpt_b, config, out: artifact)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ntempvh",
            "compare-paths",
            "--ckptA",
            "a.pt",
            "--ckptB",
            "b.pt",
            "--config",
            "path_compare.yaml",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    cli_mod.main()
    captured = capsys.readouterr()

    assert "saved path comparison:" in captured.out
    assert str(artifact) in captured.out



def test_cli_diagnostic_pipeline_reports_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact = tmp_path / "diagnostic_manifest.json"
    artifact.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli_mod, "run_diagnostic_pipeline", lambda config, out_dir=None: artifact)
    monkeypatch.setattr(
        "sys.argv",
        [
            "ntempvh",
            "diagnostic-pipeline",
            "--config",
            "diagnostic.yaml",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    cli_mod.main()
    captured = capsys.readouterr()

    assert "saved diagnostic manifest:" in captured.out
    assert str(artifact) in captured.out



def test_cli_diagnostic_pipeline_help_mentions_reusable_checkpoint_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "ntempvh",
            "diagnostic-pipeline",
            "-h",
        ],
    )

    with pytest.raises(SystemExit):
        cli_mod.main()
    captured = capsys.readouterr()

    assert "reusable checkpoint diagnostics" in captured.out
