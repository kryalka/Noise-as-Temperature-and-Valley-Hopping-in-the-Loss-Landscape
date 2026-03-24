from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def mean(values: list[Any]) -> float | None:
    numeric = [float(value) for value in values if safe_float(value) is not None]
    if not numeric:
        return None
    return float(sum(numeric) / len(numeric))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_regime_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row.get("dataset", "") or "unknown"),
            str(row.get("model", "") or "unknown"),
            str(row.get("learning_rate", "")),
            str(row.get("batch_size", "")),
        )
        grouped.setdefault(key, []).append(row)

    out_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        ok_rows = [row for row in group_rows if str(row.get("status", "")) == "ok"]
        observed_modes = sorted({str(row.get("observed_selection", "")) for row in group_rows if str(row.get("observed_selection", ""))})
        out_rows.append({
            "dataset": key[0],
            "model": key[1],
            "learning_rate": key[2],
            "batch_size": key[3],
            "num_pairs": int(len(group_rows)),
            "num_ok_pairs": int(sum(1 for row in group_rows if str(row.get("status", "")) == "ok")),
            "num_geometry_partial_pairs": int(sum(1 for row in group_rows if str(row.get("status", "")) == "geometry_partial")),
            "num_compare_failed_pairs": int(sum(1 for row in group_rows if str(row.get("status", "")) == "compare_failed")),
            "observed_selection_modes": ",".join(observed_modes),
            "mean_chord_DeltaL": mean([row.get("chord_DeltaL") for row in ok_rows]),
            "mean_observed_DeltaL": mean([row.get("observed_DeltaL") for row in ok_rows]),
            "mean_Peakobs": mean([row.get("Peakobs") for row in ok_rows]),
            "mean_Pitchord": mean([row.get("Pitchord") for row in ok_rows]),
            "mean_Pitobs": mean([row.get("Pitobs") for row in ok_rows]),
            "mean_BarrierGap": mean([row.get("BarrierGap") for row in ok_rows]),
            "mean_devL1": mean([row.get("devL1") for row in ok_rows]),
            "mean_LengthRatio": mean([row.get("LengthRatio") for row in ok_rows]),
            "mean_LengthExcess": mean([row.get("LengthExcess") for row in ok_rows]),
            "mean_curvature_proxy_A": mean([row.get("curvature_proxy_A") for row in ok_rows]),
            "mean_curvature_proxy_B": mean([row.get("curvature_proxy_B") for row in ok_rows]),
            "mean_curvature_proxy_mean": mean([row.get("curvature_proxy_mean") for row in ok_rows]),
            "mean_sigma_kappa_A": mean([row.get("sigma_kappa_A") for row in ok_rows]),
            "mean_sigma_kappa_B": mean([row.get("sigma_kappa_B") for row in ok_rows]),
            "mean_anisotropy_A": mean([row.get("anisotropy_A") for row in ok_rows]),
            "mean_anisotropy_B": mean([row.get("anisotropy_B") for row in ok_rows]),
        })
    return out_rows


def build_regime_maps(regime_rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, Any]:
    maps: dict[str, Any] = {}
    for metric_name in metric_names:
        metric_map: dict[str, Any] = {}
        for row in regime_rows:
            dataset = str(row.get("dataset", "") or "unknown")
            model = str(row.get("model", "") or "unknown")
            lr = str(row.get("learning_rate", ""))
            batch_size = str(row.get("batch_size", ""))
            metric_map.setdefault(dataset, {}).setdefault(model, {}).setdefault(lr, {})[batch_size] = safe_float(row.get(metric_name))
        maps[metric_name] = metric_map
    return maps


def build_report_markdown(
    *,
    report: dict[str, Any],
    pair_csv: Path,
    regime_csv: Path,
    maps_json: Path,
) -> str:
    counts = report["counts"]
    lines = [
        "# Diagnostic Report",
        "",
        "этот отчёт собран автоматически как reusable слой для checkpoint trajectory diagnostics",
        "",
        "## Что вошло",
        "",
        f"- config: `{report['config_path']}`",
        f"- resolved pairs csv: `{report['inputs']['resolved_pairs_csv']}`",
        f"- compare config: `{report['inputs']['compare_config']}`",
        f"- geometry config: `{report['inputs']['geometry_config']}`",
        "",
        "## Как Это Понимать",
        "",
        "- этот слой не привязан жёстко к cifar и resnet preset ам",
        "- базовый контракт, это пары checkpoint путей и совместимые compare plus geometry конфиги",
        "- если у пользователя есть свои trajectory пары, их можно передать напрямую через pairs csv",
        "",
        "## Сколько получилось",
        "",
        f"- входных пар: {counts['num_pairs_input']}",
        f"- строк в pair таблице: {counts['num_pair_rows']}",
        f"- ok пар: {counts['num_ok_pairs']}",
        f"- partial по geometry: {counts['num_geometry_partial_pairs']}",
        f"- compare failed: {counts['num_compare_failed_pairs']}",
        f"- уникальных checkpoint ов: {counts['num_unique_checkpoints']}",
        "",
        "## Основные файлы",
        "",
        f"- pair таблица: `{pair_csv}`",
        f"- regime таблица: `{regime_csv}`",
        f"- regime maps json: `{maps_json}`",
        "",
    ]
    issues = report.get("issues", [])
    if issues:
        lines.extend(["## Issues", ""])
        for issue in issues:
            lines.append(f"- {issue}")
        lines.append("")
    return "\n".join(lines)
