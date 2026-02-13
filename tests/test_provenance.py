from __future__ import annotations

from pathlib import Path

from ntempvh.utils.provenance import build_provenance




def test_build_provenance_records_core_runtime_fields(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    input_path = tmp_path / "input.csv"
    config_path.write_text("seed: 1\n", encoding="utf-8")
    input_path.write_text("x\n", encoding="utf-8")

    payload = build_provenance(
        project_root=__file__,
        config_paths=[config_path],
        input_paths=[input_path],
    )

    assert payload["created_at_utc"].endswith("Z")
    assert payload["python_version"]
    assert payload["python_executable"]
    assert payload["platform"]
    assert payload["cwd"] == str(Path.cwd().resolve())
    assert payload["config_paths"] == [str(config_path.resolve())]
    assert payload["input_paths"] == [str(input_path.resolve())]
    assert "git" in payload
