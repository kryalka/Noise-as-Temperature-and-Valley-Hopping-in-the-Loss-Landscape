from __future__ import annotations
import argparse
from pathlib import Path

from ntempvh.utils.io import load_yaml, ensure_dir
from ntempvh.train.trainer import train_one_run
from ntempvh.eval.interpolation import run_interpolation
from ntempvh.eval.barrier import compute_barrier
from ntempvh.eval.geometry import compute_geometry
from ntempvh.eval.path_compare import compare_paths
from ntempvh.pipeline.diagnostic_pipeline import run_diagnostic_pipeline

import hashlib
import json

def _short_hash(obj) -> str:
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


def main():
    ap = argparse.ArgumentParser(prog="ntempvh")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="Train one run")
    p_train.add_argument("--config", required=True)
    p_train.add_argument("--seed", type=int, required=True)
    p_train.add_argument("--out", default="outputs/runs")
    p_train.add_argument("--dry_run", action="store_true")

    p_interp = sub.add_parser("interpolate", help="Interpolate between two checkpoints")
    p_interp.add_argument("--ckptA", required=True)
    p_interp.add_argument("--ckptB", required=True)
    p_interp.add_argument("--config", required=True) 
    p_interp.add_argument("--out", default="outputs/artifacts/interp")

    p_bar = sub.add_parser("barrier", help="Compute barrier from interpolation csv")
    p_bar.add_argument("--interp_csv", required=True)
    p_bar.add_argument("--config", required=True)  
    p_bar.add_argument("--out", default="outputs/artifacts/barrier")

    p_geo = sub.add_parser("geometry", help="Compute proxy geometry (curvature) at a checkpoint")
    p_geo.add_argument("--ckpt", required=True)
    p_geo.add_argument("--config", required=True)
    p_geo.add_argument("--out", default="outputs/artifacts/geometry")

    p_compare = sub.add_parser(
        "compare-paths",
        help="Compare chord interpolation against observed training path between two checkpoints",
    )
    p_compare.add_argument("--ckptA", required=True)
    p_compare.add_argument("--ckptB", required=True)
    p_compare.add_argument("--config", required=True)
    p_compare.add_argument("--out", default="outputs/artifacts/path_compare")

    p_diag = sub.add_parser(
        "diagnostic-pipeline",
        help="Run reusable checkpoint diagnostics from trajectory pairs to summaries and regime maps",
        description="run reusable checkpoint diagnostics from trajectory pairs to summaries and regime maps",
    )
    p_diag.add_argument("--config", required=True)
    p_diag.add_argument("--out", default=None)

    args = ap.parse_args()

    if args.cmd == "train":
        cfg = load_yaml(args.config)
        train_cfg = cfg.get("training", {}) or {}

        cfg["training"] = train_cfg
        run_id = _format_run_id(cfg, args.seed)
        base_out = ensure_dir(Path(args.out))
        run_dir = ensure_dir(base_out / run_id)

        manifest = {
            "cmd": "train",
            "config_path": str(args.config),
            "seed": int(args.seed),
            "out_root": str(base_out),
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
        (run_dir / "cli_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

        if args.dry_run:
            print("DRY RUN: created run dir and manifest, training not started.")
            print(f"run dir: {run_dir}")
            return

        ckpt_path = train_one_run(cfg, seed=args.seed, out_dir=str(run_dir))
        print(f"expected metrics: {run_dir / 'metrics.jsonl'}")
        print(f"expected summary: {run_dir / 'summary.json'}")
        print(f"checkpoints dir: {run_dir / 'checkpoints'}")
        print(f"saved checkpoint: {ckpt_path}")
        print(f"run dir: {run_dir}")
        return

    if args.cmd == "interpolate":
        out_dir = str(ensure_dir(Path(args.out)))
        out_csv = run_interpolation(args.ckptA, args.ckptB, args.config, out_dir)
        print(f"saved interpolation: {out_csv}")
        return

    if args.cmd == "barrier":
        out_file = compute_barrier(args.interp_csv, args.config, args.out)
        print(f"saved barrier: {out_file}")
        return
    
    if args.cmd == "geometry":
        out_file = compute_geometry(args.ckpt, args.config, args.out)
        failed, reason = _describe_geometry_result(out_file)
        if failed:
            print(f"saved geometry failure artifact: {out_file}")
            if reason:
                print(f"geometry failure reason: {reason}")
        else:
            print(f"saved geometry: {out_file}")
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
