#!/usr/bin/env python3
from __future__ import annotations

from ntempvh.pipeline.trajectory_pairs import build_trajectory_pairs_batch

def main() -> None:
    build_trajectory_pairs_batch(
        "outputs/runs_lr_bs_grid",
        out_csv="outputs/summaries/trajectory_pairs.csv",
        out_json="outputs/summaries/trajectory_pairs_summary.json",
    )

if __name__ == "__main__":
    main()
