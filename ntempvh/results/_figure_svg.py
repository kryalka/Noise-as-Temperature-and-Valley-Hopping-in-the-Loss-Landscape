from __future__ import annotations

import html
from pathlib import Path
from typing import Any




def svg_escape(value: Any) -> str:
    return html.escape(str(value), quote=True)



def write_svg(path: Path, *, width: int, height: int, elements: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        f'{"".join(elements)}</svg>'
    )
    path.write_text(svg, encoding="utf-8")



def render_placeholder_svg(path: Path, *, title: str, message: str) -> None:
    width = 720
    height = 220
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" stroke="#d0d7de"/>',
        f'<text x="24" y="42" font-size="22" font-family="monospace" fill="#111827">{svg_escape(title)}</text>',
        f'<text x="24" y="104" font-size="16" font-family="monospace" fill="#6b7280">{svg_escape(message)}</text>',
    ]
    write_svg(path, width=width, height=height, elements=elements)



def seq_color(value: float | None, *, vmin: float, vmax: float) -> str:
    if value is None:
        return "#f3f4f6"
    if vmax <= vmin:
        ratio = 0.5
    else:
        ratio = max(0.0, min(1.0, (value - vmin) / (vmax - vmin)))
    r = int(245 - 120 * ratio)
    g = int(252 - 40 * ratio)
    b = int(245 - 170 * ratio)
    return f"rgb({r},{g},{b})"



def diverging_color(value: float | None, *, vabs: float) -> str:
    if value is None:
        return "#f3f4f6"
    if vabs <= 0.0:
        return "#f8fafc"
    ratio = max(-1.0, min(1.0, value / vabs))
    if ratio >= 0.0:
        r = 255
        g = int(245 - 120 * ratio)
        b = int(245 - 160 * ratio)
    else:
        ratio = abs(ratio)
        r = int(245 - 130 * ratio)
        g = int(248 - 90 * ratio)
        b = 255
    return f"rgb({r},{g},{b})"


def format_num(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3g}"



def render_heatmap_svg(
    path: Path,
    *,
    title: str,
    metric_name: str,
    lr_labels: list[str],
    bs_labels: list[str],
    value_map: dict[tuple[str, str], float | None],
    diverging: bool,
) -> None:
    if not lr_labels or not bs_labels:
        render_placeholder_svg(path, title=title, message=f"No data for {metric_name}")
        return

    cell_w = 90
    cell_h = 48
    left = 120
    top = 78
    width = left + len(lr_labels) * cell_w + 40
    height = top + len(bs_labels) * cell_h + 70

    values = [value for value in value_map.values() if value is not None]
    vmin = min(values) if values else 0.0
    vmax = max(values) if values else 1.0
    vabs = max(abs(vmin), abs(vmax))

    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="36" font-size="22" font-family="monospace" fill="#111827">{svg_escape(title)}</text>',
        f'<text x="24" y="58" font-size="13" font-family="monospace" fill="#6b7280">{svg_escape(metric_name)}</text>',
    ]

    for idx, lr in enumerate(lr_labels):
        x = left + idx * cell_w + cell_w / 2
        elements.append(
            f'<text x="{x:.1f}" y="{top - 16}" font-size="13" text-anchor="middle" '
            f'font-family="monospace" fill="#374151">{svg_escape(lr)}</text>'
        )

    for idx, bs in enumerate(bs_labels):
        y = top + idx * cell_h + cell_h / 2 + 5
        elements.append(
            f'<text x="{left - 12}" y="{y:.1f}" font-size="13" text-anchor="end" '
            f'font-family="monospace" fill="#374151">{svg_escape(bs)}</text>'
        )

    for yi, bs in enumerate(bs_labels):
        for xi, lr in enumerate(lr_labels):
            value = value_map.get((lr, bs))
            fill = diverging_color(value, vabs=vabs) if diverging else seq_color(value, vmin=vmin, vmax=vmax)
            x = left + xi * cell_w
            y = top + yi * cell_h
            elements.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 2}" height="{cell_h - 2}" '
                f'fill="{fill}" stroke="#d1d5db"/>'
            )
            elements.append(
                f'<text x="{x + cell_w / 2:.1f}" y="{y + cell_h / 2 + 5:.1f}" font-size="12" '
                f'text-anchor="middle" font-family="monospace" fill="#111827">{svg_escape(format_num(value))}</text>'
            )

    write_svg(path, width=width, height=height, elements=elements)



def render_multi_metric_bars_svg(
    path: Path,
    *,
    title: str,
    labels: list[str],
    metric_specs: list[tuple[str, str, str]],
    rows: list[dict[str, Any]],
) -> None:
    if not labels or not rows:
        render_placeholder_svg(path, title=title, message="No rows available")
        return

    panel_h = 210
    width = max(820, 200 + len(labels) * 90)
    height = 70 + len(metric_specs) * panel_h
    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="24" y="36" font-size="22" font-family="monospace" fill="#111827">{svg_escape(title)}</text>',
    ]

    row_map = {str(row.get("regime_label", row.get("observed_selection", ""))): row for row in rows}
    if not row_map:
        render_placeholder_svg(path, title=title, message="No usable rows available")
        return

    for panel_idx, (key, label, color) in enumerate(metric_specs):
        panel_top = 60 + panel_idx * panel_h
        chart_left = 80
        chart_top = panel_top + 20
        chart_bottom = panel_top + 150
        chart_h = chart_bottom - chart_top

        values = [None if row_map.get(item) is None else row_map[item].get(key) for item in labels]
        valid_values = [float(value) for value in values if value is not None]
        if valid_values:
            min_val = min(min(valid_values), 0.0)
            max_val = max(max(valid_values), 0.0)
            if max_val <= min_val:
                max_val = min_val + 1.0
        else:
            min_val = 0.0
            max_val = 1.0

        zero_ratio = (0.0 - min_val) / (max_val - min_val)
        zero_y = chart_bottom - zero_ratio * chart_h

        elements.append(
            f'<text x="24" y="{panel_top + 22}" font-size="14" font-family="monospace" fill="#374151">{svg_escape(label)}</text>'
        )
        elements.append(
            f'<line x1="{chart_left}" y1="{zero_y:.1f}" x2="{width - 32}" y2="{zero_y:.1f}" stroke="#9ca3af"/>'
        )

        for idx, item in enumerate(labels):
            x = chart_left + idx * 90 + 24
            value = None if values[idx] is None else float(values[idx])
            bar_h = 0.0 if value is None else abs(value / (max_val - min_val)) * chart_h
            if value is None:
                y = zero_y
                fill = "#e5e7eb"
            elif value >= 0.0:
                y = zero_y - bar_h
                fill = color
            else:
                y = zero_y
                fill = color
            text_y = (y - 6) if value is not None and value >= 0.0 else (y + bar_h + 16)
            elements.append(
                f'<rect x="{x}" y="{y:.1f}" width="42" height="{bar_h:.1f}" fill="{fill}" opacity="0.9"/>'
            )
            elements.append(
                f'<text x="{x + 21:.1f}" y="{text_y:.1f}" '
                f'font-size="11" text-anchor="middle" font-family="monospace" fill="#111827">{svg_escape(format_num(value))}</text>'
            )
            elements.append(
                f'<text x="{x + 21:.1f}" y="{chart_bottom + 18}" font-size="10" text-anchor="middle" '
                f'font-family="monospace" fill="#4b5563" transform="rotate(35 {x + 21:.1f},{chart_bottom + 18})">{svg_escape(item)}</text>'
            )

    write_svg(path, width=width, height=height, elements=elements)
