from __future__ import annotations

import csv
import json
from pathlib import Path

from ntempvh.eval.intervention_geometry import SUMMARY_COLUMNS as INTERVENTION_GEOMETRY_INPUT_COLUMNS



def write_compare_artifact(
    *,
    root: Path,
    run_dir: Path,
    with_endpoint_meta: bool,
) -> Path:
    interp_dir = root / "interpolation"
    barrier_dir = root / "barrier"
    compare_dir = root / "comparisons"
    interp_dir.mkdir(parents=True, exist_ok=True)
    barrier_dir.mkdir(parents=True, exist_ok=True)
    compare_dir.mkdir(parents=True, exist_ok=True)

    ckpt_a = run_dir / "checkpoints" / "epoch_001.pt"
    ckpt_b = run_dir / "checkpoints" / "epoch_003.pt"
    chord_interp_csv = interp_dir / "interp__chord.csv"
    observed_interp_csv = interp_dir / "interp__observed.csv"
    chord_barrier_json = barrier_dir / "barrier__chord.json"
    observed_barrier_json = barrier_dir / "barrier__observed.json"

    chord_interp_csv.write_text("t,val_loss,val_acc\n0,1,0.5\n1,0.8,0.7\n", encoding="utf-8")
    observed_interp_csv.write_text("t,val_loss,val_acc\n0,1,0.5\n1,0.7,0.75\n", encoding="utf-8")
    chord_barrier_json.write_text("{}", encoding="utf-8")
    observed_barrier_json.write_text("{}", encoding="utf-8")

    if with_endpoint_meta:
        meta = {
            "ckptA": str(ckpt_a),
            "ckptB": str(ckpt_b),
            "endpoint_eval": {
                "A": {"loss": 0.9, "acc": 0.55},
                "B": {"loss": 0.6, "acc": 0.75},
            },
        }
        chord_interp_csv.with_suffix(".meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    payload = {
        "ckptA": str(ckpt_a),
        "ckptB": str(ckpt_b),
        "pair_tag": "lr0.2__bs8__seed1__run_deadbeef__e001_e003",
        "config": {
            "path": {"num_points": 5},
            "evaluation": {"split": "val"},
        },
        "artifacts": {
            "chord_interp_csv": str(chord_interp_csv),
            "chord_meta_json": str(chord_interp_csv.with_suffix(".meta.json")),
            "chord_barrier_json": str(chord_barrier_json),
            "observed_interp_csv": str(observed_interp_csv),
            "observed_barrier_json": str(observed_barrier_json),
        },
        "metrics": {
            "chord_DeltaL": 0.1,
            "observed_DeltaL": 0.05,
            "barrier_gap": -0.05,
            "chord_length": 2.0,
            "observed_length": 3.5,
            "length_ratio": 1.75,
            "length_excess": 1.5,
            "loss_profile_l1_mean": 0.02,
            "loss_profile_linf": 0.08,
        },
        "observed_path": {"selection": "all"},
        "report_metrics": {
            "Peakobs": 0.0,
            "Pitchord": 0.0,
            "Pitobs": 0.3,
            "BarrierGap": -0.05,
            "LengthRatio": 1.75,
            "LengthExcess": 1.5,
            "devL1": 0.02,
        },
    }

    comparison_json = compare_dir / "pathcompare__sample.json"
    comparison_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return comparison_json



def write_intervention_geometry_summary(
    path: Path,
    *,
    run_dir: Path,
    run_name: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "seed": 1,
            "learning_rate": 0.2,
            "batch_size": 8,
            "intervention_start_epoch": 2,
            "intervention_end_epoch": 3,
            "intervention_lr_multiplier": 2.0,
            "intervention_batch_size": 4,
            "checkpoint_role": "pre",
            "checkpoint_path": str(run_dir / "checkpoints" / "epoch_001.pt"),
            "checkpoint_epoch": 1,
            "geometry_json": "pre.json",
            "status": "ok",
            "reason": "",
            "kappa_tr": 1.0,
            "kappa_tr_std": 0.1,
            "base_loss": 0.9,
            "base_acc": 0.55,
        },
        {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "seed": 1,
            "learning_rate": 0.2,
            "batch_size": 8,
            "intervention_start_epoch": 2,
            "intervention_end_epoch": 3,
            "intervention_lr_multiplier": 2.0,
            "intervention_batch_size": 4,
            "checkpoint_role": "post",
            "checkpoint_path": str(run_dir / "checkpoints" / "epoch_003.pt"),
            "checkpoint_epoch": 3,
            "geometry_json": "post.json",
            "status": "ok",
            "reason": "",
            "kappa_tr": 1.5,
            "kappa_tr_std": 0.1,
            "base_loss": 0.7,
            "base_acc": 0.7,
        },
        {
            "run_dir": str(run_dir),
            "run_name": run_name,
            "seed": 1,
            "learning_rate": 0.2,
            "batch_size": 8,
            "intervention_start_epoch": 2,
            "intervention_end_epoch": 3,
            "intervention_lr_multiplier": 2.0,
            "intervention_batch_size": 4,
            "checkpoint_role": "final",
            "checkpoint_path": str(run_dir / "checkpoints" / "final.pt"),
            "checkpoint_epoch": 4,
            "geometry_json": "final.json",
            "status": "ok",
            "reason": "",
            "kappa_tr": 2.0,
            "kappa_tr_std": 0.1,
            "base_loss": 0.4,
            "base_acc": 0.85,
        },
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INTERVENTION_GEOMETRY_INPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
