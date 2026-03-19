from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from ntempvh.utils.artifacts import (
    build_geometry_artifact_context,
    build_path_compare_artifact_context,
    load_json_artifact,
)

from ._diagnostic_inputs import safe_float


def geometry_json_candidates(
    ckpt_path: str | Path,
    geometry_cfg: dict[str, Any],
    *,
    geometry_root: Path,
) -> tuple[Path, Path]:
    success = build_geometry_artifact_context(ckpt_path, geometry_cfg, failed=False)
    failure = build_geometry_artifact_context(ckpt_path, geometry_cfg, failed=True)
    return geometry_root / f"{success['stem']}.json", geometry_root / f"{failure['stem']}.json"



def ensure_geometry_payload(
    ckpt_path: str | Path,
    *,
    geometry_cfg_path: Path,
    geometry_cfg: dict[str, Any],
    geometry_root: Path,
    reuse_existing: bool,
    cache: dict[str, dict[str, Any]],
    geometry_fn: Callable[[str, str, str], Path],
) -> dict[str, Any]:
    key = str(Path(ckpt_path).resolve())
    if key in cache:
        return cache[key]

    success_json, failure_json = geometry_json_candidates(
        ckpt_path,
        geometry_cfg,
        geometry_root=geometry_root,
    )

    if reuse_existing and success_json.exists():
        payload = load_json_artifact(success_json)
        payload["_artifact_path"] = str(success_json)
        cache[key] = payload
        return payload
    if reuse_existing and failure_json.exists():
        payload = load_json_artifact(failure_json)
        payload["_artifact_path"] = str(failure_json)
        cache[key] = payload
        return payload

    out_json = geometry_fn(str(ckpt_path), str(geometry_cfg_path), str(geometry_root))
    payload = load_json_artifact(out_json)
    payload["_artifact_path"] = str(out_json)
    cache[key] = payload
    return payload


def compare_json_path(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    compare_cfg: dict[str, Any],
    *,
    compare_root: Path,
) -> Path:
    artifact = build_path_compare_artifact_context(ckpt_a, ckpt_b, compare_cfg)
    return compare_root / "comparisons" / f"{artifact['stem']}.json"



def ensure_compare_payload(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    *,
    compare_cfg_path: Path,
    compare_cfg: dict[str, Any],
    compare_root: Path,
    reuse_existing: bool,
    compare_fn: Callable[[str, str, str, str], Path],
) -> dict[str, Any]:
    target_json = compare_json_path(ckpt_a, ckpt_b, compare_cfg, compare_root=compare_root)
    if reuse_existing and target_json.exists():
        payload = load_json_artifact(target_json)
        payload["_artifact_path"] = str(target_json)
        return payload

    out_json = compare_fn(str(ckpt_a), str(ckpt_b), str(compare_cfg_path), str(compare_root))
    payload = load_json_artifact(out_json)
    payload["_artifact_path"] = str(out_json)
    return payload



def diagnostic_row(
    pair_meta: dict[str, Any],
    *,
    compare_payload: dict[str, Any] | None,
    geometry_a: dict[str, Any] | None,
    geometry_b: dict[str, Any] | None,
    reason: str | None,
) -> dict[str, Any]:
    status = "ok"
    if compare_payload is None:
        status = "compare_failed"
    elif str((geometry_a or {}).get("status", "")).lower() == "failed" or str((geometry_b or {}).get("status", "")).lower() == "failed":
        status = "geometry_partial"

    report_metrics = dict((compare_payload or {}).get("report_metrics", {}) or {})
    metrics = dict((compare_payload or {}).get("metrics", {}) or {})
    compare_config = dict((compare_payload or {}).get("config", {}) or {})

    kappa_a = safe_float((geometry_a or {}).get("kappa_tr"))
    kappa_b = safe_float((geometry_b or {}).get("kappa_tr"))
    numeric_kappas = [value for value in [kappa_a, kappa_b] if value is not None]
    curvature_mean = float(sum(numeric_kappas) / len(numeric_kappas)) if numeric_kappas else None

    return {
        "comparison_json": str((compare_payload or {}).get("_artifact_path", "")),
        "status": status,
        "reason": reason or "",
        "run_name": str(pair_meta.get("run_name", "")),
        "pair_tag": str(pair_meta.get("pair_tag", "")),
        "dataset": str(pair_meta.get("dataset", "")),
        "model": str(pair_meta.get("model", "")),
        "seed": pair_meta.get("seed", ""),
        "learning_rate": pair_meta.get("learning_rate", ""),
        "batch_size": pair_meta.get("batch_size", ""),
        "epoch_A": pair_meta.get("epoch_A", ""),
        "epoch_B": pair_meta.get("epoch_B", ""),
        "ckptA": str(pair_meta.get("ckptA", "")),
        "ckptB": str(pair_meta.get("ckptB", "")),
        "observed_selection": str((((compare_payload or {}).get("observed_path") or {}).get("selection", ""))),
        "eval_split": str((((compare_config.get("evaluation") or {}).get("split", "")))),
        "chord_DeltaL": metrics.get("chord_DeltaL", ""),
        "observed_DeltaL": metrics.get("observed_DeltaL", ""),
        "Peakobs": report_metrics.get("Peakobs", ""),
        "Pitchord": report_metrics.get("Pitchord", ""),
        "Pitobs": report_metrics.get("Pitobs", ""),
        "BarrierGap": report_metrics.get("BarrierGap", ""),
        "devL1": report_metrics.get("devL1", ""),
        "LengthRatio": report_metrics.get("LengthRatio", metrics.get("length_ratio", "")),
        "LengthExcess": report_metrics.get("LengthExcess", metrics.get("length_excess", "")),
        "chord_length": metrics.get("chord_length", ""),
        "observed_length": metrics.get("observed_length", ""),
        "geometry_A_json": str((geometry_a or {}).get("_artifact_path", "")),
        "geometry_B_json": str((geometry_b or {}).get("_artifact_path", "")),
        "curvature_proxy_A": kappa_a,
        "curvature_proxy_B": kappa_b,
        "curvature_proxy_mean": curvature_mean,
        "sigma_kappa_A": safe_float((geometry_a or {}).get("sigma_kappa")),
        "sigma_kappa_B": safe_float((geometry_b or {}).get("sigma_kappa")),
        "anisotropy_A": safe_float((geometry_a or {}).get("anisotropy")),
        "anisotropy_B": safe_float((geometry_b or {}).get("anisotropy")),
    }
