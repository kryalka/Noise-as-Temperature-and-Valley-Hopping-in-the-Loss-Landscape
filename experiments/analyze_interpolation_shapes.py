#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from _interpolation_shape_support import (
    SUMMARY_COLUMNS,
    build_shape_row,
    build_shape_summary,
    print_shape_summary,
)
from ntempvh.utils.artifacts import load_interpolation_metadata

INTERP_ROOT = Path("outputs/artifacts/interpolation_trajectory")
OUT_CSV = Path("outputs/summaries/interpolation_shapes_summary.csv")
OUT_JSON = Path("outputs/summaries/interpolation_shapes_summary.json")


def main() -> None:
    csv_paths = sorted(INTERP_ROOT.glob("interp__*.csv"))

    rows: list[dict[str, Any]] = []
    bad_files: list[dict[str, str]] = []

    for csv_path in csv_paths:
        meta_path = csv_path.with_suffix(".meta.json")
        if not meta_path.exists():
            bad_files.append({
                "interp_csv": str(csv_path),
                "error": f"Missing meta json: {meta_path}",
            })
            continue

        try:
            meta_path, meta = load_interpolation_metadata(csv_path)
            df = pd.read_csv(csv_path)
        except Exception as e:
            bad_files.append({
                "interp_csv": str(csv_path),
                "error": repr(e),
            })
            continue

        required_cols = {"t", "val_loss", "val_acc"}
        if not required_cols.issubset(df.columns):
            bad_files.append({
                "interp_csv": str(csv_path),
                "error": f"Missing required columns: {required_cols - set(df.columns)}",
            })
            continue

        df = df.sort_values("t").reset_index(drop=True)
        rows.append(
            build_shape_row(
                csv_path=csv_path,
                meta_path=meta_path,
                meta=meta,
                df=df,
            )
        )

    out_df = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    summary = build_shape_summary(
        interp_root=INTERP_ROOT,
        out_csv=OUT_CSV,
        out_df=out_df,
        bad_files=bad_files,
    )

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print_shape_summary(out_csv=OUT_CSV, out_json=OUT_JSON, out_df=out_df, bad_files=bad_files)


if __name__ == "__main__":
    main()
