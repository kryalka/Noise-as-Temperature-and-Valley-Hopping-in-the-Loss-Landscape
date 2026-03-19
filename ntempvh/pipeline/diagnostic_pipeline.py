from __future__ import annotations

import argparse
from pathlib import Path

from ntempvh.eval.geometry import compute_geometry
from ntempvh.eval.path_compare import compare_paths
from ntempvh.results.diagnostic_outputs import write_diagnostic_outputs
from ntempvh.utils.io import ensure_dir, load_yaml

from ._diagnostic_inputs import (
    ensure_mapping,
    resolve_pair_input,
    resolve_path,
    validate_diagnostic_config,
)
from ._diagnostic_runtime import (
    diagnostic_row,
    ensure_compare_payload,
    ensure_geometry_payload,
)



def run_diagnostic_pipeline(
    config_path: str | Path,
    *,
    out_dir: str | Path | None = None,
) -> Path:
    config_path = Path(config_path).resolve()
    cfg = load_yaml(config_path)
    validate_diagnostic_config(cfg)

    base_dir = config_path.parent
    outputs_cfg = ensure_mapping("outputs", cfg.get("outputs"))
    diagnostics_cfg = ensure_mapping("diagnostics", cfg.get("diagnostics"))

    out_root = resolve_path(out_dir or outputs_cfg["out_root"], base_dir=base_dir)
    ensure_dir(out_root)

    compare_cfg_path = resolve_path(str(diagnostics_cfg["compare_config"]), base_dir=base_dir)
    geometry_cfg_path = resolve_path(str(diagnostics_cfg["geometry_config"]), base_dir=base_dir)
    compare_cfg = load_yaml(compare_cfg_path)
    geometry_cfg = load_yaml(geometry_cfg_path)
    reuse_existing = bool(diagnostics_cfg.get("reuse_existing", True))

    compare_root = ensure_dir(out_root / "compare")
    geometry_root = ensure_dir(out_root / "geometry")

    pair_rows, pair_input_meta = resolve_pair_input(cfg, config_path=config_path, out_root=out_root)
    geometry_cache: dict[str, dict[str, object]] = {}
    issues: list[str] = []
    diagnostic_rows: list[dict[str, object]] = []

    for pair_meta in pair_rows:
        compare_payload: dict[str, object] | None = None
        compare_reason: str | None = None
        try:
            compare_payload = ensure_compare_payload(
                pair_meta["ckptA"],
                pair_meta["ckptB"],
                compare_cfg_path=compare_cfg_path,
                compare_cfg=compare_cfg,
                compare_root=compare_root,
                reuse_existing=reuse_existing,
                compare_fn=compare_paths,
            )
        except Exception as exc:
            compare_reason = str(exc)
            issues.append(f"compare failed for {pair_meta.get('pair_tag', '')}: {exc}")

        geometry_a = ensure_geometry_payload(
            pair_meta["ckptA"],
            geometry_cfg_path=geometry_cfg_path,
            geometry_cfg=geometry_cfg,
            geometry_root=geometry_root,
            reuse_existing=reuse_existing,
            cache=geometry_cache,
            geometry_fn=compute_geometry,
        )
        geometry_b = ensure_geometry_payload(
            pair_meta["ckptB"],
            geometry_cfg_path=geometry_cfg_path,
            geometry_cfg=geometry_cfg,
            geometry_root=geometry_root,
            reuse_existing=reuse_existing,
            cache=geometry_cache,
            geometry_fn=compute_geometry,
        )

        for failed_payload, ckpt_key in ((geometry_a, "ckptA"), (geometry_b, "ckptB")):
            if str(failed_payload.get("status", "")).lower() == "failed":
                issues.append(
                    f"geometry failed for {pair_meta[ckpt_key]}: {failed_payload.get('reason', 'unknown_reason')}"
                )

        diagnostic_rows.append(
            diagnostic_row(
                pair_meta,
                compare_payload=compare_payload,
                geometry_a=geometry_a,
                geometry_b=geometry_b,
                reason=compare_reason,
            )
        )

    outputs = write_diagnostic_outputs(
        rows=diagnostic_rows,
        out_root=out_root,
        config_path=config_path,
        input_meta={
            "resolved_pairs_csv": str(out_root / "resolved_pairs.csv"),
            "compare_config": str(compare_cfg_path),
            "geometry_config": str(geometry_cfg_path),
            "reuse_existing": reuse_existing,
            "input_mode": pair_input_meta.get("input_mode", ""),
            "reusable_scope": [
                "works with any checkpoint pairs that match the compare and geometry configs",
                "preset cifar and resnet flows are only reference entrypoints",
                "external trajectory pairs can be passed directly through pairs_csv",
            ],
        },
        issues=issues,
        unique_checkpoint_count=len(geometry_cache),
    )

    print(f"Сохранены пары диагностики: {outputs['pair_csv']}")
    print(f"Сохранена таблица режимов: {outputs['regime_csv']}")
    print(f"Сохранены карты режимов : {outputs['maps_json']}")
    print(f"Сохранён отчёт          : {outputs['report_md']}")
    print(f"Сохранён manifest       : {outputs['manifest_json']}")
    return outputs["manifest_json"]


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ntempvh.pipeline.diagnostic_pipeline",
        description="run reusable checkpoint trajectory diagnostics from pair inputs to summary tables and regime maps",
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_diagnostic_pipeline(args.config, out_dir=args.out)


if __name__ == "__main__":
    main()
