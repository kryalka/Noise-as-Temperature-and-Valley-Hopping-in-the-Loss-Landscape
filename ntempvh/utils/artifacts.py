from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ntempvh.utils.checkpoints import build_checkpoint_tag, build_pair_tag
from ntempvh.utils.config_validation import (
    validate_barrier_config,
    validate_geometry_config,
    validate_interpolation_config,
    validate_path_compare_config,
)


def _sanitize_tag(value: Any) -> str:
    text = str(value).strip().lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in text)
    cleaned = cleaned.strip("_")
    return cleaned or "default"



def _normalize_for_signature(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _normalize_for_signature(val)
            for key, val in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_for_signature(item) for item in value]
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def build_config_signature(payload: Any, *, length: int = 10) -> str:
    normalized = _normalize_for_signature(payload)
    blob = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:length]


def load_json_artifact(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def interpolation_meta_path(path: str | Path) -> Path:
    path = Path(path)
    if path.name.endswith(".meta.json"):
        return path
    return path.with_suffix(".meta.json")



def load_interpolation_metadata(path: str | Path) -> tuple[Path, dict[str, Any]]:
    meta_path = interpolation_meta_path(path)
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing interpolation metadata: {meta_path}")

    meta = load_json_artifact(meta_path)
    if not isinstance(meta, dict):
        raise ValueError(f"Invalid interpolation metadata in {meta_path}: expected a JSON object")
    if "ckptA" not in meta or "ckptB" not in meta:
        raise ValueError(
            f"Invalid interpolation metadata in {meta_path}: missing 'ckptA' or 'ckptB'"
        )
    return meta_path, meta



def _normalize_observed_path_config(path_cfg: dict[str, Any]) -> dict[str, Any]:
    observed_cfg = path_cfg.get("observed", {}) if isinstance(path_cfg, dict) else {}
    observed_cfg = observed_cfg or {}
    return {
        "selection": str(observed_cfg.get("selection", "all")).strip().lower(),
        "milestone_epochs": [int(epoch) for epoch in (observed_cfg.get("milestone_epochs", []) or [])],
        "epochs": [int(epoch) for epoch in (observed_cfg.get("epochs", []) or [])],
    }



def build_interpolation_artifact_context(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = cfg or {}
    validate_interpolation_config(cfg)
    path_cfg = cfg.get("path", {}) if isinstance(cfg, dict) else {}
    path_cfg = path_cfg or {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    eval_cfg = eval_cfg or {}

    config = {
        "data_root": str(cfg.get("data_root", "./data")),
        "path": {
            "type": str(path_cfg.get("type", "linear")).strip().lower(),
            "num_points": int(path_cfg.get("num_points", cfg.get("num_points", 41))),
            "bn_recalib_batches": int(
                path_cfg.get("bn_recalib_batches", cfg.get("bn_recalib_batches", 0))
            ),
            "pivots": [str(p) for p in (path_cfg.get("pivots", []) or [])],
        },
        "evaluation": {
            "split": str(eval_cfg.get("split", cfg.get("split", "val"))).strip().lower(),
            "batch_size": int(eval_cfg.get("batch_size", cfg.get("batch_size", 256))),
            "bn_batch_size": int(
                eval_cfg.get(
                    "bn_batch_size",
                    eval_cfg.get("batch_size", cfg.get("batch_size", 256)),
                )
            ),
            "val_size": int(eval_cfg.get("val_size", 5000)),
            "split_seed": int(eval_cfg.get("split_seed", 0)),
        },
    }
    if config["path"]["type"] == "observed" or "observed" in path_cfg:
        config["path"]["observed"] = _normalize_observed_path_config(path_cfg)

    pair_tag = build_pair_tag(ckpt_a, ckpt_b)
    config_signature = build_config_signature(config)
    stem_parts = [
        f"interp__{pair_tag}",
        _sanitize_tag(config["path"]["type"]),
    ]
    if config["path"]["type"] == "observed":
        stem_parts.append(
            f"sel_{_sanitize_tag(config['path']['observed']['selection'])}"
        )
    stem_parts.extend(
        [
            f"n{config['path']['num_points']}",
            _sanitize_tag(config["evaluation"]["split"]),
            f"cfg_{config_signature}",
        ]
    )
    stem = "__".join(stem_parts)
    return {
        "pair_tag": pair_tag,
        "config": config,
        "config_signature": config_signature,
        "stem": stem,
    }



def build_barrier_artifact_context(interp_csv: str | Path, cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = cfg or {}
    validate_barrier_config(cfg)
    barrier_cfg = cfg.get("barrier", {}) if isinstance(cfg, dict) else {}
    barrier_cfg = barrier_cfg or {}

    config = {
        "definition": str(barrier_cfg.get("definition", "max_minus_endpoints")).strip().lower(),
        "thresholds": [
            float(x)
            for x in barrier_cfg.get("thresholds", [0.005, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5])
        ],
    }
    interp_stem = Path(interp_csv).stem
    interp_tag = interp_stem[len("interp__") :] if interp_stem.startswith("interp__") else interp_stem
    config_signature = build_config_signature(config)
    stem = (
        f"barrier__{interp_tag}"
        f"__{_sanitize_tag(config['definition'])}"
        f"__cfg_{config_signature}"
    )
    return {
        "interp_stem": interp_stem,
        "interp_tag": interp_tag,
        "config": config,
        "config_signature": config_signature,
        "stem": stem,
    }



def build_path_compare_artifact_context(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    cfg: dict[str, Any] | None,
) -> dict[str, Any]:
    cfg = cfg or {}
    validate_path_compare_config(cfg)

    path_cfg = cfg.get("path", {}) if isinstance(cfg, dict) else {}
    path_cfg = path_cfg or {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    eval_cfg = eval_cfg or {}
    barrier_cfg = cfg.get("barrier", {}) if isinstance(cfg, dict) else {}
    barrier_cfg = barrier_cfg or {}

    config = {
        "data_root": str(cfg.get("data_root", "./data")),
        "path": {
            "num_points": int(path_cfg.get("num_points", cfg.get("num_points", 41))),
            "bn_recalib_batches": int(
                path_cfg.get("bn_recalib_batches", cfg.get("bn_recalib_batches", 0))
            ),
            "observed": _normalize_observed_path_config(path_cfg),
        },
        "evaluation": {
            "split": str(eval_cfg.get("split", cfg.get("split", "val"))).strip().lower(),
            "batch_size": int(eval_cfg.get("batch_size", cfg.get("batch_size", 256))),
            "bn_batch_size": int(
                eval_cfg.get(
                    "bn_batch_size",
                    eval_cfg.get("batch_size", cfg.get("batch_size", 256)),
                )
            ),
            "val_size": int(eval_cfg.get("val_size", 5000)),
            "split_seed": int(eval_cfg.get("split_seed", 0)),
        },
        "barrier": {
            "definition": str(barrier_cfg.get("definition", "max_minus_endpoints")).strip().lower(),
            "thresholds": [
                float(x)
                for x in barrier_cfg.get("thresholds", [0.005, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5])
            ],
        },
    }

    pair_tag = build_pair_tag(ckpt_a, ckpt_b)
    config_signature = build_config_signature(config)
    stem = "__".join(
        [
            f"pathcompare__{pair_tag}",
            f"sel_{_sanitize_tag(config['path']['observed']['selection'])}",
            f"n{config['path']['num_points']}",
            f"cfg_{config_signature}",
        ]
    )
    return {
        "pair_tag": pair_tag,
        "config": config,
        "config_signature": config_signature,
        "stem": stem,
    }



def build_geometry_artifact_context(
    ckpt_path: str | Path,
    cfg: dict[str, Any] | None,
    *,
    failed: bool = False,
) -> dict[str, Any]:
    cfg = cfg or {}
    validate_geometry_config(cfg)
    geometry_cfg = cfg.get("geometry", {}) if isinstance(cfg, dict) else {}
    geometry_cfg = geometry_cfg or {}
    eval_cfg = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    eval_cfg = eval_cfg or {}

    num_eval_batches = geometry_cfg.get("num_eval_batches", cfg.get("num_eval_batches", None))
    num_eval_batches = int(num_eval_batches) if num_eval_batches is not None else None

    config = {
        "data_root": str(cfg.get("data_root", "./data")),
        "geometry": {
            "alpha": float(geometry_cfg.get("alpha", cfg.get("alpha", 1e-3))),
            "num_directions": int(
                geometry_cfg.get("num_directions", cfg.get("num_directions", 10))
            ),
            "eval_batch_size": int(
                geometry_cfg.get("eval_batch_size", cfg.get("eval_batch_size", 256))
            ),
            "num_eval_batches": num_eval_batches,
            "bn_recalib_batches": int(
                geometry_cfg.get("bn_recalib_batches", cfg.get("bn_recalib_batches", 0))
            ),
        },
        "evaluation": {
            "val_size": int(eval_cfg.get("val_size", 5000)),
            "split_seed": int(eval_cfg.get("split_seed", 0)),
        },
    }

    checkpoint_tag = build_checkpoint_tag(ckpt_path)
    config_signature = build_config_signature(config)
    prefix = "geometry_failed" if failed else "geometry"
    stem = f"{prefix}__{checkpoint_tag}__cfg_{config_signature}"
    return {
        "checkpoint_tag": checkpoint_tag,
        "config": config,
        "config_signature": config_signature,
        "stem": stem,
    }
