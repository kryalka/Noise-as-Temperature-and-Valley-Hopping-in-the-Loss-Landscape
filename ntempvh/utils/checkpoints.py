from __future__ import annotations

import hashlib
from pathlib import Path
import re
from typing import Any

RUN_RE = re.compile(
    r"""
    seed(?P<seed>\d+)
    __opt(?P<optimizer>[^_]+)
    _lr(?P<lr>[^_]+)
    _bs(?P<bs>[^_]+)
    _wd(?P<wd>[^_]+)
    _mom(?P<mom>[^_]+)
    _sch(?P<scheduler>[^_]+)
    __(?P<hash>[a-f0-9]+)
    """,
    re.VERBOSE,
)

RUN_NAME_RE = re.compile(
    r"""
    ^
    (?P<dataset>[^_]+)_
    (?P<model>[^_]+)_
    seed(?P<seed>\d+)
    __opt(?P<optimizer>[^_]+)
    _lr(?P<lr>[^_]+)
    _bs(?P<bs>[^_]+)
    _wd(?P<wd>[^_]+)
    _mom(?P<mom>[^_]+)
    _sch(?P<scheduler>[^_]+)
    __(?P<hash>[a-f0-9]+)
    $
    """,
    re.VERBOSE,
)

EPOCH_RE = re.compile(r"^epoch_(\d+)\.pt$")


def _lr_to_str(x: float) -> str:
    return f"{x:g}"


def _short_path_hash(value: str | Path, *, length: int = 8) -> str:
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:length]



def parse_run_name(run_name: str) -> dict[str, Any]:
    m = RUN_NAME_RE.match(str(run_name))
    if m is None:
        return {}

    g = m.groupdict()
    return {
        "run_name": str(run_name),
        "dataset": g["dataset"],
        "model": g["model"],
        "seed": int(g["seed"]),
        "learning_rate": float(g["lr"]),
        "batch_size": int(g["bs"]),
        "run_hash": g["hash"],
        "optimizer": g["optimizer"],
        "weight_decay": float(g["wd"]),
        "momentum": float(g["mom"]),
        "scheduler": g["scheduler"],
    }



def parse_checkpoint_path(ckpt_path: str | Path) -> dict:
    p = Path(ckpt_path)
    run_name = p.parent.parent.name

    em = EPOCH_RE.match(p.name)
    if em is None:
        raise ValueError(
            "Could not parse checkpoint epoch from path "
            f"'{ckpt_path}': file name '{p.name}' does not match 'epoch_XXX.pt'"
        )

    parsed = parse_run_name(run_name)
    if not parsed:
        m = RUN_RE.search(run_name)
        if m is None:
            raise ValueError(
                "Could not parse checkpoint run metadata from path "
                f"'{ckpt_path}': run directory '{run_name}' does not match the expected naming pattern"
            )

        g = m.groupdict()
        parsed = {
            "run_name": run_name,
            "seed": int(g["seed"]),
            "learning_rate": float(g["lr"]),
            "batch_size": int(g["bs"]),
            "run_hash": g["hash"],
            "optimizer": g["optimizer"],
            "weight_decay": float(g["wd"]),
            "momentum": float(g["mom"]),
            "scheduler": g["scheduler"],
        }

    parsed["epoch"] = int(em.group(1))
    return parsed



def collect_epoch_checkpoints(checkpoints_dir: str | Path) -> list[tuple[int, Path]]:
    checkpoints_dir = Path(checkpoints_dir)
    pairs: list[tuple[int, Path]] = []
    for path in checkpoints_dir.iterdir():
        if not path.is_file():
            continue
        m = EPOCH_RE.match(path.name)
        if m is None:
            continue
        pairs.append((int(m.group(1)), path))

    pairs.sort(key=lambda item: item[0])
    return pairs



def resolve_observed_checkpoint_sequence(
    ckpt_a: str | Path,
    ckpt_b: str | Path,
    *,
    selection: str = "all",
    milestone_epochs: list[int] | tuple[int, ...] | None = None,
    epochs: list[int] | tuple[int, ...] | None = None,
) -> list[Path]:
    info_a = parse_checkpoint_path(ckpt_a)
    info_b = parse_checkpoint_path(ckpt_b)

    run_dir_a = Path(ckpt_a).resolve().parent.parent
    run_dir_b = Path(ckpt_b).resolve().parent.parent
    if info_a["run_name"] != info_b["run_name"] or run_dir_a != run_dir_b:
        raise ValueError(
            "Observed path is supported only within a single run, got:\n"
            f"  A: {ckpt_a}\n"
            f"  B: {ckpt_b}"
        )

    checkpoints_dir = run_dir_a / "checkpoints"
    epoch_to_path = {epoch: path for epoch, path in collect_epoch_checkpoints(checkpoints_dir)}
    if info_a["epoch"] not in epoch_to_path:
        raise FileNotFoundError(f"Endpoint checkpoint epoch {info_a['epoch']} not found in {checkpoints_dir}")
    if info_b["epoch"] not in epoch_to_path:
        raise FileNotFoundError(f"Endpoint checkpoint epoch {info_b['epoch']} not found in {checkpoints_dir}")

    epoch_a = int(info_a["epoch"])
    epoch_b = int(info_b["epoch"])
    lo = min(epoch_a, epoch_b)
    hi = max(epoch_a, epoch_b)
    direction = 1 if epoch_b >= epoch_a else -1

    available_epochs = sorted(epoch for epoch in epoch_to_path.keys() if lo <= epoch <= hi)

    sel = str(selection).strip().lower()
    milestone_epochs = [int(epoch) for epoch in (milestone_epochs or [])]
    epochs = [int(epoch) for epoch in (epochs or [])]

    if sel == "all":
        selected_epochs = list(available_epochs)
    elif sel == "milestones":
        selected_epochs = [epoch for epoch in milestone_epochs if lo <= epoch <= hi and epoch in epoch_to_path]
    elif sel == "explicit":
        outside_epochs = [epoch for epoch in epochs if epoch < lo or epoch > hi]
        if outside_epochs:
            raise ValueError(
                f"Explicit observed epochs must lie within [{lo}, {hi}], got {outside_epochs}"
            )

        missing_epochs = [epoch for epoch in epochs if epoch not in epoch_to_path]
        if missing_epochs:
            raise FileNotFoundError(
                f"Explicit observed epochs are missing checkpoint files in {checkpoints_dir}: {missing_epochs}"
            )
        selected_epochs = list(epochs)
    else:
        raise ValueError(f"Unknown observed checkpoint selection mode: {selection}")

    selected_epochs = sorted(set([epoch_a, *selected_epochs, epoch_b]))
    if direction < 0:
        selected_epochs = list(reversed(selected_epochs))

    return [epoch_to_path[epoch] for epoch in selected_epochs]



def build_pair_tag(ckpt_a: str | Path, ckpt_b: str | Path) -> str:
    a = parse_checkpoint_path(ckpt_a)
    b = parse_checkpoint_path(ckpt_b)

    if a["run_name"] != b["run_name"]:
        raise ValueError(
            f"Checkpoint pair is expected within a single run, got:\\n"
            f"  A: {ckpt_a}\\n"
            f"  B: {ckpt_b}"
        )

    return (
        f"lr{_lr_to_str(a['learning_rate'])}"
        f"__bs{a['batch_size']}"
        f"__seed{a['seed']}"
        f"__run{a['run_hash']}"
        f"__e{a['epoch']:03d}_e{b['epoch']:03d}"
    )



def build_checkpoint_tag(ckpt_path: str | Path) -> str:
    try:
        info = parse_checkpoint_path(ckpt_path)
    except ValueError:
        p = Path(ckpt_path)
        return f"{p.stem}__{_short_path_hash(p)}"

    return (
        f"lr{_lr_to_str(info['learning_rate'])}"
        f"__bs{info['batch_size']}"
        f"__seed{info['seed']}"
        f"__run{info['run_hash']}"
        f"__e{info['epoch']:03d}"
    )



def validate_checkpoint_pair(ckpt_a: dict, ckpt_b: dict) -> None:
    model_a = str(ckpt_a.get("model", "")).lower()
    model_b = str(ckpt_b.get("model", "")).lower()
    if model_a != model_b:
        raise ValueError(f"Checkpoint model mismatch: {model_a} vs {model_b}")

    dataset_a = str(ckpt_a.get("dataset", "")).lower()
    dataset_b = str(ckpt_b.get("dataset", "")).lower()
    if dataset_a != dataset_b:
        raise ValueError(f"Checkpoint dataset mismatch: {dataset_a} vs {dataset_b}")

    state_dict_a = ckpt_a["state_dict"]
    state_dict_b = ckpt_b["state_dict"]

    if list(state_dict_a.keys()) != list(state_dict_b.keys()):
        raise ValueError("State dict keys do not match between checkpoints")

    for key in state_dict_a.keys():
        if state_dict_a[key].shape != state_dict_b[key].shape:
            raise ValueError(
                f"Shape mismatch for key {key}: {state_dict_a[key].shape} vs {state_dict_b[key].shape}"
            )
