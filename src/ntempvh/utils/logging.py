from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from .io import append_jsonl, ensure_dir


@dataclass
class RunLogger:
    out_dir: Path

    def __post_init__(self) -> None:
        ensure_dir(self.out_dir)

    @property
    def metrics_path(self) -> Path:
        return self.out_dir / "metrics.jsonl"

    def log(self, record: Dict[str, Any]) -> None:
        append_jsonl(self.metrics_path, record)
