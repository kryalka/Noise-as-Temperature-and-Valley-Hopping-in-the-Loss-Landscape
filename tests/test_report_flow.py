from __future__ import annotations

import csv
import json
from pathlib import Path

from ntempvh.pipeline.trajectory_pairs import build_trajectory_pairs_batch
from ntempvh.utils.io import load_yaml

from ._report_flow_support import run_name, write_run_stub



def test_build_trajectory_pairs_batch_supports_custom_runs_root(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs_cifar100"
    name = run_name(dataset="cifar100", seed=1, lr=0.2, bs=64)
    write_run_stub(
        runs_root=runs_root,
        run_name=name,
        dataset="cifar100",
        epochs=[1, 10, 20],
    )

    out_csv = tmp_path / "pairs.csv"
    out_json = tmp_path / "pairs_summary.json"
    summary_json = build_trajectory_pairs_batch(
        runs_root,
        out_csv=out_csv,
        out_json=out_json,
        pair_mode="milestones",
        milestone_epochs=[1, 10, 20],
    )

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))

    assert summary["num_runs"] == 1
    assert summary["num_pairs"] == 2
    assert summary["pairs_by_dataset"]["cifar100"] == 2
    assert len(rows) == 2
    assert rows[0]["dataset"] == "cifar100"
    assert rows[0]["epoch_A"] == "1"
    assert rows[0]["epoch_B"] == "10"



def test_build_trajectory_pairs_batch_supports_explicit_pairs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    name = run_name(dataset="cifar10", seed=1, lr=0.2, bs=64)
    write_run_stub(
        runs_root=runs_root,
        run_name=name,
        dataset="cifar10",
        epochs=[1, 10, 50, 70, 80, 100],
    )

    out_csv = tmp_path / "pairs_explicit.csv"
    out_json = tmp_path / "pairs_explicit_summary.json"
    build_trajectory_pairs_batch(
        runs_root,
        out_csv=out_csv,
        out_json=out_json,
        pair_mode="explicit_pairs",
        explicit_pairs=["50:100", "70:100", "80:100"],
    )

    rows = list(csv.DictReader(out_csv.open(encoding="utf-8")))
    summary = json.loads(out_json.read_text(encoding="utf-8"))

    assert [(row["epoch_A"], row["epoch_B"]) for row in rows] == [("50", "100"), ("70", "100"), ("80", "100")]
    assert summary["pair_mode"] == "explicit_pairs"
    assert summary["explicit_pairs"] == [[50, 100], [70, 100], [80, 100]]



def test_report_presets_encode_report_specific_pairs_and_test_eval() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    report_cifar10 = load_yaml(repo_root / "configs/pipeline/report_cifar10.yaml")
    report_cifar100 = load_yaml(repo_root / "configs/pipeline/report_cifar100.yaml")
    report_cifar10_resnet34 = load_yaml(repo_root / "configs/pipeline/report_cifar10_resnet34.yaml")
    report_cifar100_resnet34 = load_yaml(repo_root / "configs/pipeline/report_cifar100_resnet34.yaml")
    interpolation_test_cfg = load_yaml(repo_root / "configs/eval/interpolation_test.yaml")
    compare_test_cfg = load_yaml(repo_root / "configs/eval/path_compare_test.yaml")
    geometry_cfg = load_yaml(repo_root / "configs/eval/geometry.yaml")
    intervention_report_cfg = load_yaml(repo_root / "configs/train/intervention_lr_bs_grid_report.yaml")
    intervention_report_cifar100_cfg = load_yaml(
        repo_root / "configs/train/intervention_lr_bs_grid_report_cifar100.yaml"
    )

    late_pairs = {"50:100", "70:100", "80:100"}
    expected_windows = {(50, 100), (60, 65), (65, 70), (70, 75), (75, 80), (80, 85)}
    expected_multipliers = {0.7, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.75}

    cifar10_pairs = set(report_cifar10["baseline"]["trajectory_pairs"]["explicit_pairs"])
    cifar100_pairs = set(report_cifar100["baseline"]["trajectory_pairs"]["explicit_pairs"])
    cifar10_resnet34_pairs = set(report_cifar10_resnet34["baseline"]["trajectory_pairs"]["explicit_pairs"])
    cifar100_resnet34_pairs = set(report_cifar100_resnet34["baseline"]["trajectory_pairs"]["explicit_pairs"])
    report_variants = intervention_report_cfg["intervention_variants"]
    report_cifar100_variants = intervention_report_cifar100_cfg["intervention_variants"]
    report_windows = {(row["start_epoch"], row["end_epoch"]) for row in report_variants}
    report_cifar100_windows = {(row["start_epoch"], row["end_epoch"]) for row in report_cifar100_variants}
    report_multipliers = {float(row["lr_multiplier"]) for row in report_variants}
    report_cifar100_multipliers = {float(row["lr_multiplier"]) for row in report_cifar100_variants}

    assert report_cifar10["baseline"]["trajectory_pairs"]["pair_mode"] == "explicit_pairs"
    assert report_cifar100["baseline"]["trajectory_pairs"]["pair_mode"] == "explicit_pairs"
    assert late_pairs.issubset(cifar10_pairs)
    assert late_pairs.issubset(cifar100_pairs)
    assert late_pairs.issubset(cifar10_resnet34_pairs)
    assert late_pairs.issubset(cifar100_resnet34_pairs)
    assert report_cifar10["baseline"]["interpolation_config"].endswith("interpolation_test.yaml")
    assert report_cifar100["baseline"]["interpolation_config"].endswith("interpolation_test.yaml")
    assert report_cifar10_resnet34["baseline"]["interpolation_config"].endswith("interpolation_test.yaml")
    assert report_cifar100_resnet34["baseline"]["interpolation_config"].endswith("interpolation_test.yaml")
    assert report_cifar10["baseline"]["path_compare_config"].endswith("path_compare_test.yaml")
    assert report_cifar100["baseline"]["path_compare_config"].endswith("path_compare_test.yaml")
    assert report_cifar10_resnet34["baseline"]["path_compare_config"].endswith("path_compare_test.yaml")
    assert report_cifar100_resnet34["baseline"]["path_compare_config"].endswith("path_compare_test.yaml")
    assert report_cifar10["intervention"]["train_grid_config"].endswith("intervention_lr_bs_grid_report.yaml")
    assert report_cifar100["intervention"]["train_grid_config"].endswith("intervention_lr_bs_grid_report_cifar100.yaml")
    assert interpolation_test_cfg["evaluation"]["split"] == "test"
    assert compare_test_cfg["evaluation"]["split"] == "test"
    assert geometry_cfg["geometry"]["num_directions"] == 100
    assert report_windows == expected_windows
    assert report_cifar100_windows == expected_windows
    assert report_multipliers == expected_multipliers
    assert report_cifar100_multipliers == expected_multipliers
    assert len(report_variants) == 48
    assert len(report_cifar100_variants) == 48



def test_build_stage_plan_supports_repo_custom_report_config() -> None:
    import ntempvh.pipeline.report_flow as report_mod

    project_root = Path(__file__).resolve().parents[1]
    cfg = load_yaml(project_root / "configs/pipeline/report_custom_example.yaml")

    stages, outputs = report_mod._build_stage_plan(
        cfg,
        project_root=project_root,
        python_bin="python3.11",
    )
    by_stage = {stage["stage"]: stage for stage in stages}
    trajectory_cmd = by_stage["trajectory_pairs"]["command"]

    assert "configs/train/lr_bs_grid_custom_example.yaml" in by_stage["baseline_train"]["command"]
    assert "configs/train/intervention_lr_bs_grid_custom_example.yaml" in by_stage["intervention_train"]["command"]
    assert "configs/eval/path_compare_custom_example.yaml" in by_stage["path_compare"]["command"]
    assert trajectory_cmd[trajectory_cmd.index("--pair_mode") + 1] == "milestones"
    for epoch in ["1", "5", "10", "20", "40", "50"]:
        assert epoch in trajectory_cmd

    assert outputs["baseline_runs_root"].endswith("outputs/runs_lr_bs_grid_custom_example")
    assert outputs["intervention_runs_root"].endswith("outputs/runs_lr_bs_grid_intervention_custom_example")
    assert outputs["figure_outputs_out"].endswith("outputs/summaries/figure_outputs_custom_example")
