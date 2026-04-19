"""CLI entry points for single runs and small diagnostic jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from ntempvh.eval.barrier import compute_barrier
from ntempvh.eval.geometry import compute_geometry
from ntempvh.eval.interpolation import run_interpolation
from ntempvh.eval.path_compare import compare_paths
from ntempvh.pipeline.diagnostic_pipeline import run_diagnostic_pipeline
from ntempvh.train.trainer import train_one_run
from ntempvh.utils.io import ensure_dir, load_yaml


def _short_hash(obj: object) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def _active_intervention_fingerprint(cfg: dict) -> dict | None:
    intervention = cfg.get("intervention", {}) or {}
    if not bool(intervention.get("enabled", False)):
        return None
    return {
        "enabled": True,
        "start_epoch": intervention.get("start_epoch"),
        "end_epoch": intervention.get("end_epoch"),
        "lr_multiplier": intervention.get("lr_multiplier"),
        "batch_size": (
            None if intervention.get("batch_size", None) is None else intervention.get("batch_size")
        ),
    }


def _format_run_id(cfg: dict, seed: int) -> str:
    tr = cfg.get("training", {}) or {}
    dataset = str(cfg.get("dataset", "data")).lower()
    model = str(cfg.get("model", "model")).lower()

    opt = str(tr.get("optimizer", "sgd")).lower()
    lr = tr.get("learning_rate", "na")
    bs = tr.get("batch_size", "na")
    wd = tr.get("weight_decay", "na")
    mom = tr.get("momentum", "na")
    sch = str(tr.get("scheduler", "none")).lower()

    fingerprint = {
        "dataset": dataset,
        "model": model,
        "training": tr,
        "data_root": cfg.get("data_root"),
        "data": cfg.get("data", {}),
        "seed": int(seed),
    }
    intervention_fp = _active_intervention_fingerprint(cfg)
    if intervention_fp is not None:
        fingerprint["intervention"] = intervention_fp
    h = _short_hash(fingerprint)

    return f"{dataset}_{model}_seed{seed}__opt{opt}_lr{lr}_bs{bs}_wd{wd}_mom{mom}_sch{sch}__{h}"


def _read_json_if_possible(path: Path) -> dict | None:
    if path.suffix != ".json" or not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _describe_geometry_result(path: str | Path) -> tuple[bool, str | None]:
    path = Path(path)
    payload = _read_json_if_possible(path)
    if isinstance(payload, dict) and str(payload.get("status", "")).lower() == "failed":
        reason = payload.get("reason")
        return True, str(reason) if reason not in (None, "") else None
    if path.stem.startswith("geometry_failed__"):
        return True, None
    return False, None


def _write_train_manifest(
    *,
    run_dir: Path,
    config_path: str,
    seed: int,
    out_root: Path,
    run_id: str,
    cfg: dict,
) -> None:
    manifest = {
        "cmd": "train",
        "config_path": str(config_path),
        "seed": int(seed),
        "out_root": str(out_root),
        "run_id": str(run_id),
        "run_dir": str(run_dir),
        "cfg_fingerprint": {
            "dataset": str(cfg.get("dataset", "")),
            "model": str(cfg.get("model", "")),
            "training": cfg.get("training", {}),
            "data_root": cfg.get("data_root"),
            "data": cfg.get("data", {}),
            "intervention": _active_intervention_fingerprint(cfg),
        },
    }
    (run_dir / "cli_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _handle_train_command(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config)
    cfg["training"] = cfg.get("training", {}) or {}

    run_id = _format_run_id(cfg, args.seed)
    out_root = ensure_dir(Path(args.out))
    run_dir = ensure_dir(out_root / run_id)
    _write_train_manifest(
        run_dir=run_dir,
        config_path=args.config,
        seed=args.seed,
        out_root=out_root,
        run_id=run_id,
        cfg=cfg,
    )

    if args.dry_run:
        print("dry run: created run directory and manifest")
        print(f"run directory: {run_dir}")
        return

    ckpt_path = train_one_run(cfg, seed=args.seed, out_dir=str(run_dir))
    print(f"metrics: {run_dir / 'metrics.jsonl'}")
    print(f"summary: {run_dir / 'summary.json'}")
    print(f"checkpoints: {run_dir / 'checkpoints'}")
    print(f"checkpoint: {ckpt_path}")
    print(f"run directory: {run_dir}")


def _handle_geometry_command(args: argparse.Namespace) -> None:
    out_file = compute_geometry(args.ckpt, args.config, args.out)
    failed, reason = _describe_geometry_result(out_file)
    if failed:
        print(f"saved geometry failure artifact: {out_file}")
        if reason:
            print(f"geometry failure reason: {reason}")
        return
    print(f"geometry json: {out_file}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="ntempvh",
        description="CLI for single-run training and evaluation",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train one run")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--out", default="outputs/runs")
    p_train.add_argument("--dry_run", action="store_true")

    p_interp = sub.add_parser("interpolate", help="Evaluate interpolation for one checkpoint pair")
    p_interp.add_argument("--ckptA", required=True)
    p_interp.add_argument("--ckptB", required=True)
    p_interp.add_argument("--config", required=True)
    p_interp.add_argument("--out", default="outputs/artifacts/interp")

    p_bar = sub.add_parser("barrier", help="Compute barrier statistics from an interpolation CSV")
    p_bar.add_argument("--interp_csv", required=True)
    p_bar.add_argument("--config", required=True)
    p_bar.add_argument("--out", default="outputs/artifacts/barrier")

    p_geo = sub.add_parser("geometry", help="Estimate local geometry for one checkpoint")
    p_geo.add_argument("--ckpt", required=True)
    p_geo.add_argument("--config", required=True)
    p_geo.add_argument("--out", default="outputs/artifacts/geometry")

    p_compare = sub.add_parser(
        "compare-paths",
        help="Compare chord and observed paths for one checkpoint pair",
    )
    p_compare.add_argument("--ckptA", required=True)
    p_compare.add_argument("--ckptB", required=True)
    p_compare.add_argument("--config", required=True)
    p_compare.add_argument("--out", default="outputs/artifacts/path_compare")

    p_diag = sub.add_parser(
        "diagnostic-pipeline",
        help="Run checkpoint diagnostics for explicit pairs or a pairs CSV",
        description="Run checkpoint diagnostics for explicit pairs or a pairs CSV",
    )
    p_diag.add_argument("--config", required=True)
    p_diag.add_argument("--out", default=None)
    return ap


def main() -> None:
    args = build_parser().parse_args()
    if args.cmd == "train":
        _handle_train_command(args)
        return

    if args.cmd == "interpolate":
        out_dir = str(ensure_dir(Path(args.out)))
        out_csv = run_interpolation(args.ckptA, args.ckptB, args.config, out_dir)
        print(f"interpolation csv: {out_csv}")
        return

    if args.cmd == "barrier":
        out_file = compute_barrier(args.interp_csv, args.config, args.out)
        print(f"barrier json: {out_file}")
        return

    if args.cmd == "geometry":
        _handle_geometry_command(args)
        return

    if args.cmd == "compare-paths":
        out_file = compare_paths(args.ckptA, args.ckptB, args.config, args.out)
        print(f"saved path comparison: {out_file}")
        return

    if args.cmd == "diagnostic-pipeline":
        out_file = run_diagnostic_pipeline(args.config, out_dir=args.out)
        print(f"saved diagnostic manifest: {out_file}")
        return


if __name__ == "__main__":
    main()
