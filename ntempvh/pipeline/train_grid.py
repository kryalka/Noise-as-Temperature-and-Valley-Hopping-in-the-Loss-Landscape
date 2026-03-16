from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import yaml

from ntempvh.cli import _format_run_id
from ntempvh.utils.config_validation import validate_train_config
from ntempvh.utils.io import load_yaml

from ._train_grid_helpers import (
    build_train_variant_cfg,
    expand_intervention_variants,
    fmt_lr_for_filename,
    load_train_grid_config,
    path_for_cli,
    variant_tag,
)


@dataclass
class TrainGridJob:
    seed: int
    learning_rate: float
    batch_size: int
    cfg_path: str
    out_root: str
    run_dir: str


def build_train_grid_jobs(
    grid_config_path: str | Path,
    *,
    tmp_cfg_dir: str | Path,
    project_root: str | Path | None = None,
) -> list[TrainGridJob]:
    grid_config_path = Path(grid_config_path).resolve()
    project_root = Path(project_root or Path.cwd()).resolve()
    tmp_cfg_dir = Path(tmp_cfg_dir).resolve()
    tmp_cfg_dir.mkdir(parents=True, exist_ok=True)

    grid_cfg = load_train_grid_config(grid_config_path)
    base_config_path = Path(str(grid_cfg["base_config"]))
    if not base_config_path.is_absolute():
        base_config_path = (project_root / base_config_path).resolve()

    base_cfg = load_yaml(base_config_path)
    out_root = str(grid_cfg.get("out_root", "outputs/runs_lr_bs_grid"))
    seeds = [int(value) for value in grid_cfg["seeds"]]
    learning_rates = [float(value) for value in grid_cfg["learning_rates"]]
    batch_sizes = [int(value) for value in grid_cfg["batch_sizes"]]
    config_overrides = dict(grid_cfg.get("config_overrides", {}) or {})
    intervention_variants = expand_intervention_variants(grid_cfg)

    jobs: list[TrainGridJob] = []
    for variant_idx, intervention_variant in enumerate(intervention_variants):
        tag = variant_tag(intervention_variant, idx=variant_idx)
        for learning_rate in learning_rates:
            for batch_size in batch_sizes:
                cfg = build_train_variant_cfg(
                    base_cfg,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                    config_overrides=config_overrides,
                    intervention_variant=intervention_variant,
                )
                validate_train_config(cfg)

                filename = f"train_lr{fmt_lr_for_filename(learning_rate)}_bs{batch_size}"
                if tag:
                    filename = f"{filename}__{tag}"

                tmp_cfg_path = tmp_cfg_dir / f"{filename}.yaml"
                with open(tmp_cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

                for seed in seeds:
                    run_id = _format_run_id(cfg, int(seed))
                    jobs.append(
                        TrainGridJob(
                            seed=int(seed),
                            learning_rate=float(learning_rate),
                            batch_size=int(batch_size),
                            cfg_path=path_for_cli(tmp_cfg_path, project_root=project_root),
                            out_root=str(out_root),
                            run_dir=str(Path(out_root) / run_id),
                        )
                    )

    return jobs


def print_train_grid_jobs(
    grid_config_path: str | Path,
    *,
    tmp_cfg_dir: str | Path,
    project_root: str | Path | None = None,
) -> None:
    jobs = build_train_grid_jobs(
        grid_config_path,
        tmp_cfg_dir=tmp_cfg_dir,
        project_root=project_root,
    )
    for job in jobs:
        print("\t".join([str(job.seed), str(job.learning_rate), str(job.batch_size), job.cfg_path, job.out_root, job.run_dir]))


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m ntempvh.pipeline.train_grid",
        description="Expand a train-grid config into concrete run jobs",
    )
    ap.add_argument("--grid_config", required=True)
    ap.add_argument("--tmp_cfg_dir", required=True)
    ap.add_argument("--project_root", default=str(Path.cwd().resolve()))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    jobs = build_train_grid_jobs(
        args.grid_config,
        tmp_cfg_dir=args.tmp_cfg_dir,
        project_root=args.project_root,
    )
    if args.json:
        print(json.dumps([job.__dict__ for job in jobs], ensure_ascii=False, indent=2))
        return

    for job in jobs:
        print("\t".join([str(job.seed), str(job.learning_rate), str(job.batch_size), job.cfg_path, job.out_root, job.run_dir]))


if __name__ == "__main__":
    main()
