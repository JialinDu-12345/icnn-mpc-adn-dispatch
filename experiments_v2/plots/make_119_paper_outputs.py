from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_v2.config import RESULT_DIR              


DEFAULT_FIG_DIR = RESULT_DIR / "paper_figures" / "119"
DEFAULT_TABLE_DIR = RESULT_DIR / "paper_tables" / "119"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images"

DISPATCH_METHODS = ["Proposed-H3", "SAC", "MPC-3"]
SCALABILITY_METHODS = ["MPC-3", "SAC-ICNN", "SAC", "Proposed-H3", "Perfect-Global"]

METHOD_LABELS = {
    "Proposed-H3": "Proposed-H3",
    "SAC": "SAC",
    "MPC-3": "MPC-3",
}

SCHEDULE_METHOD_TO_FILE = {
    "Proposed-H3": "scheduling_results_119_P_pro.pdf",
    "SAC": "scheduling_results_119_P_sac.pdf",
    "MPC-3": "scheduling_results_119_P_mpc.pdf",
}

SCHEDULE_COLORS = {
    "Load": "#222222",
    "SOC": "#7A5195",
    "DG": "#4C78A8",
    "WT": "#54A24B",
    "PV": "#ECA82C",
    "Grid": "#9D755D",
    "ESS": "#E45756",
    "Loss": "#BAB0AC",
}


def setup_matplotlib() -> None:
    rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 11,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else RESULT_DIR / raw


def ensure_dirs(*paths: Optional[Path]) -> None:
    for path in paths:
        if path is not None:
            path.mkdir(parents=True, exist_ok=True)


def bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def completed_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "completed" not in df.columns:
        return df.copy()
    return df[bool_series(df["completed"])].copy()


def numeric_value(value: object) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.nan
    return number if math.isfinite(number) else math.nan


def mean_ci_text(mean_value: object, ci_value: object, digits: int) -> str:
    mean_number = numeric_value(mean_value)
    ci_number = numeric_value(ci_value)
    if not math.isfinite(mean_number):
        return "-"
    if not math.isfinite(ci_number):
        return f"{mean_number:.{digits}f}"
    return f"{mean_number:.{digits}f} +/- {ci_number:.{digits}f}"


def per_step_ci(row: pd.Series) -> float:
    std_value = numeric_value(row.get("std_avg_step_time_s"))
    count = numeric_value(row.get("n_completed"))
    if not math.isfinite(std_value) or not math.isfinite(count) or count < 2:
        return math.nan
    return 1.96 * std_value / math.sqrt(count)


def voltage_text(row: pd.Series) -> str:
    value = numeric_value(row.get("mean_cost_constraint"))
    if not math.isfinite(value) or abs(value) <= 5e-10:
        return "0"
    return f"{value:.4f}"


def scalability_rows(input_path: Path) -> list[dict[str, str]]:
    df = pd.read_csv(input_path, encoding="utf-8-sig")
    rows: list[dict[str, str]] = []
    for method in SCALABILITY_METHODS:
        selected = df[df["method"].astype(str).eq(method)]
        if selected.empty:
            raise RuntimeError(f"Missing 119-bus scalability row for {method}.")
        row = selected.iloc[0]
        rows.append(
            {
                "Method": method,
                "Cost ($)": mean_ci_text(row.get("mean_cost"), row.get("ci95_cost"), 2),
                "Gap (%)": "-"
                if method == "Perfect-Global"
                else mean_ci_text(
                    row.get("mean_gap_to_perfect_percent"),
                    row.get("ci95_gap_to_perfect_percent"),
                    2,
                ),
                "Per-step time (s)": "-"
                if method == "Perfect-Global"
                else mean_ci_text(
                    row.get("mean_avg_step_time_s"),
                    per_step_ci(row),
                    3,
                ),
                "Vio. (p.u.)": voltage_text(row),
            }
        )
    return rows


def write_scalability_tables(input_name: str, table_dir: Path) -> None:
    input_path = resolve_path(input_name)
    if not input_path.exists():
        print("[SKIP] 119-bus scalability table input is not ready yet:", input_path)
        return
    rows = scalability_rows(input_path)
    ensure_dirs(table_dir)
    frame = pd.DataFrame(rows)
    csv_path = table_dir / "table_119_scalability.csv"
    frame.to_csv(csv_path, index=False, encoding="utf-8")
    print("Saved:", csv_path)

    headers = list(frame.columns)
    md_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if index == 0 else "---:" for index in range(len(headers))) + " |",
    ]
    for row in rows:
        md_lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    md_path = table_dir / "table_119_scalability.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("Saved:", md_path)

    tex_lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Method & Cost (\$) & Gap (\%) & Per-step time (s) & Vio. (p.u.) \\",
        r"\midrule",
    ]
    for row in rows:
        tex_values = [row[header].replace("+/-", r"$\pm$") for header in headers]
        tex_lines.append(" & ".join(tex_values) + r" \\")
    tex_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex_path = table_dir / "table_119_scalability.tex"
    tex_path.write_text("\n".join(tex_lines), encoding="utf-8")
    print("Saved:", tex_path)


def savefig(fig: plt.Figure, filename: str, figure_dir: Path, image_dir: Optional[Path]) -> None:
    ensure_dirs(figure_dir, image_dir)
    fig_path = figure_dir / filename
    fig.savefig(fig_path, format="pdf", bbox_inches="tight")
    print("Saved:", fig_path)
    if image_dir is not None:
        image_path = image_dir / filename
        shutil.copyfile(fig_path, image_path)
        print("Saved:", image_path)


def parse_json_list(value: object, n_steps: int = 48) -> np.ndarray:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return np.full(n_steps, np.nan, dtype=float)
    text = str(value).strip()
    if not text:
        return np.full(n_steps, np.nan, dtype=float)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return np.full(n_steps, np.nan, dtype=float)
    values = np.asarray(data, dtype=float).reshape(-1)
    if values.size >= n_steps:
        return values[:n_steps]
    out = np.full(n_steps, np.nan, dtype=float)
    out[: values.size] = values
    return out


def bar_values(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def dispatch_ylim(*arrays: np.ndarray) -> tuple[float, float]:
    clean = []
    for values in arrays:
        finite = values[np.isfinite(values)]
        if finite.size:
            clean.extend(finite.tolist())
    if not clean:
        return -2.0, 8.0
    ymin = min(clean)
    ymax = max(clean)
    span = max(ymax - ymin, 1.0)
    return min(-0.5, ymin - 0.12 * span), ymax + 0.18 * span


def secondary_ylim_align_zero(
    *,
    left_ymin: float,
    left_ymax: float,
    right_ymax: float = 105.0,
    zero_value: float = 0.0,
) -> tuple[float, float]:
    if left_ymax <= left_ymin:
        return 0.0, right_ymax

    zero_frac = (zero_value - left_ymin) / (left_ymax - left_ymin)
    zero_frac = min(max(zero_frac, 1e-6), 1.0 - 1e-6)
    right_ymin = -zero_frac * right_ymax / (1.0 - zero_frac)
    return right_ymin, right_ymax


def method_label_for_schedule(method: str) -> str:
    labels = {
        "Proposed-H3": "(a)",
        "SAC": "(b)",
        "MPC-3": "(c)",
    }
    return labels.get(method, "")


def max_abs_finite(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    return float(np.max(np.abs(finite)))


def signed_grid_for_plot(
    *,
    method: str,
    load: np.ndarray,
    p_dg: np.ndarray,
    p_bss: np.ndarray,
    p_grid: np.ndarray,
    loss: np.ndarray,
    wt: np.ndarray,
    pv: np.ndarray,
) -> np.ndarray:
    implied_grid = load + loss - p_dg - wt - pv - p_bss
    raw_balance_error = max_abs_finite(
        p_dg + wt + pv + p_grid + p_bss - loss - load
    )
    implied_balance_error = max_abs_finite(
        p_dg + wt + pv + implied_grid + p_bss - loss - load
    )
    if raw_balance_error > max(1e-4, implied_balance_error + 1e-4):
        repaired = np.where(np.isfinite(implied_grid), implied_grid, p_grid)
        print(
            f"[FIX] {method}: using balance-implied signed Grid "
            f"for plotting (raw balance error {raw_balance_error:.3g} MW)."
        )
        return repaired
    return p_grid


def dispatch_series_from_row(row: pd.Series) -> dict[str, object]:
    n_steps = 48
    method = str(row["method"])
    load = parse_json_list(row.get("step_load_total_json"), n_steps)
    soc_mwh = parse_json_list(row.get("step_SOC_mean_json"), n_steps)
    p_dg = parse_json_list(row.get("step_P_DG_total_json"), n_steps)
    p_bss = parse_json_list(row.get("step_P_BSS_json"), n_steps)
    p_grid_raw = parse_json_list(row.get("step_P_grid_json"), n_steps)
    loss = parse_json_list(row.get("step_loss_json"), n_steps)
    wt = parse_json_list(row.get("step_wt_total_json"), n_steps)
    pv = parse_json_list(row.get("step_pv_total_json"), n_steps)
    p_grid = signed_grid_for_plot(
        method=method,
        load=load,
        p_dg=p_dg,
        p_bss=p_bss,
        p_grid=p_grid_raw,
        loss=loss,
        wt=wt,
        pv=pv,
    )
    return {
        "method": method,
        "load": load,
        "soc_mwh": soc_mwh,
        "p_dg": p_dg,
        "p_bss": p_bss,
        "p_grid": p_grid,
        "loss": loss,
        "wt": wt,
        "pv": pv,
    }


def dispatch_axis_arrays(series: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    load = series["load"]
    p_dg = series["p_dg"]
    p_bss = series["p_bss"]
    p_grid = series["p_grid"]
    loss = series["loss"]
    wt = series["wt"]
    pv = series["pv"]

    assert isinstance(load, np.ndarray)
    assert isinstance(p_dg, np.ndarray)
    assert isinstance(p_bss, np.ndarray)
    assert isinstance(p_grid, np.ndarray)
    assert isinstance(loss, np.ndarray)
    assert isinstance(wt, np.ndarray)
    assert isinstance(pv, np.ndarray)

    ess_pos = np.maximum(p_bss, 0.0)
    ess_neg = np.minimum(p_bss, 0.0)
    grid_pos = np.maximum(p_grid, 0.0)
    grid_neg = np.minimum(p_grid, 0.0)
    loss_neg = -np.maximum(loss, 0.0)
    positive_stack = p_dg + wt + pv + grid_pos + ess_pos
    negative_stack = grid_neg + ess_neg + loss_neg
    return load, positive_stack, negative_stack


def dispatch_data_range(*arrays: np.ndarray) -> tuple[float, float]:
    values = []
    for array in arrays:
        finite = array[np.isfinite(array)]
        if finite.size:
            values.extend(finite.tolist())
    if not values:
        return -2.0, 8.0
    return min(values), max(values)


def nice_tick_step(raw_step: float) -> float:
    if not np.isfinite(raw_step) or raw_step <= 0:
        return 1.0
    magnitude = 10.0 ** math.floor(math.log10(raw_step))
    fraction = raw_step / magnitude
    for multiplier in [1.0, 2.0, 2.5, 4.0, 5.0, 10.0]:
        if fraction <= multiplier:
            return multiplier * magnitude
    return 10.0 * magnitude


def common_dispatch_axis(
    series_list: list[dict[str, object]],
) -> tuple[tuple[float, float], np.ndarray]:
    arrays = []
    for series in series_list:
        arrays.extend(dispatch_axis_arrays(series))
    data_ymin, data_ymax = dispatch_data_range(*arrays)
    span = max(data_ymax - data_ymin, 1.0)
    padded_ymin = min(-0.5, data_ymin - 0.20 * span)
    padded_ymax = data_ymax + 0.08 * span
    step = nice_tick_step((padded_ymax - padded_ymin) / 8.0)
    tick_min = math.floor((padded_ymin + 1e-9) / step) * step
    tick_max = math.ceil((padded_ymax - 1e-9) / step) * step
    ticks = np.arange(tick_min, tick_max + 0.5 * step, step)
    ticks[np.isclose(ticks, 0.0)] = 0.0
    print(
        "Common Active power axis:",
        f"ylim=({tick_min:g}, {tick_max:g}),",
        "ticks=" + ",".join(f"{tick:g}" for tick in ticks),
    )
    return (float(tick_min), float(tick_max)), ticks


def plot_single_dispatch_schedule(
    series: dict[str, object],
    *,
    left_ylim: tuple[float, float],
    left_yticks: np.ndarray,
    figure_dir: Path,
    image_dir: Optional[Path],
) -> None:
    method = str(series["method"])
    output_name = SCHEDULE_METHOD_TO_FILE.get(method)
    if output_name is None:
        return

    n_steps = 48
    x = np.arange(n_steps) / 2.0
    load = series["load"]
    soc_mwh = series["soc_mwh"]
    p_dg = series["p_dg"]
    p_bss = series["p_bss"]
    p_grid = series["p_grid"]
    loss = series["loss"]
    wt = series["wt"]
    pv = series["pv"]

    assert isinstance(load, np.ndarray)
    assert isinstance(soc_mwh, np.ndarray)
    assert isinstance(p_dg, np.ndarray)
    assert isinstance(p_bss, np.ndarray)
    assert isinstance(p_grid, np.ndarray)
    assert isinstance(loss, np.ndarray)
    assert isinstance(wt, np.ndarray)
    assert isinstance(pv, np.ndarray)

    # Physical SOC uses the 2.5 MWh rated capacity: 0.125-2.375 MWh = 5-95%.
    # For 119 buses, soc_mwh is the mean energy of four equal-capacity ESSs.
    ess_capacity_mwh = 2.5
    soc_percent = 100.0 * soc_mwh / ess_capacity_mwh
    soc_x = np.concatenate(([-0.5], x))
    soc_percent_plot = np.concatenate(([50.0], soc_percent))

    ess_pos = np.maximum(p_bss, 0.0)
    ess_neg = np.minimum(p_bss, 0.0)
    grid_pos = np.maximum(p_grid, 0.0)
    grid_neg = np.minimum(p_grid, 0.0)
    loss_neg = -np.maximum(loss, 0.0)

    fig, ax1 = plt.subplots(figsize=(9.0, 4.0))
    width = 0.42
    bottom_pos = np.zeros(n_steps)
    bar_handles = {}

    for values, label in [
        (p_dg, "DG"),
        (wt, "WT"),
        (pv, "PV"),
        (grid_pos, "Grid"),
        (ess_pos, "ESS"),
    ]:
        heights = bar_values(values)
        handle = ax1.bar(
            x,
            heights,
            width=width,
            bottom=bottom_pos,
            color=SCHEDULE_COLORS[label],
            edgecolor="none",
            linewidth=0.0,
            label=label,
            zorder=2,
        )
        bottom_pos = bottom_pos + heights
        bar_handles[label] = handle

    bottom_neg = np.zeros(n_steps)
    grid_neg_handle = ax1.bar(
        x,
        bar_values(grid_neg),
        width=width,
        bottom=bottom_neg,
        color=SCHEDULE_COLORS["Grid"],
        edgecolor="none",
        linewidth=0.0,
        label="Grid",
        zorder=2,
    )
    bottom_neg = bottom_neg + bar_values(grid_neg)
    ess_neg_handle = ax1.bar(
        x,
        bar_values(ess_neg),
        width=width,
        bottom=bottom_neg,
        color=SCHEDULE_COLORS["ESS"],
        edgecolor="none",
        linewidth=0.0,
        label="ESS",
        zorder=2,
    )
    bottom_neg = bottom_neg + bar_values(ess_neg)
    loss_handle = ax1.bar(
        x,
        bar_values(loss_neg),
        width=width,
        bottom=bottom_neg,
        color=SCHEDULE_COLORS["Loss"],
        edgecolor="none",
        linewidth=0.0,
        label="Loss",
        zorder=2,
    )

    if "Grid" not in bar_handles:
        bar_handles["Grid"] = grid_neg_handle
    if "ESS" not in bar_handles:
        bar_handles["ESS"] = ess_neg_handle

    (load_line,) = ax1.plot(
        x,
        load,
        color=SCHEDULE_COLORS["Load"],
        marker="*",
        markersize=6.5,
        linewidth=1.6,
        label="Load",
        zorder=4,
    )

    ax1.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    left_ymin, left_ymax = left_ylim
    ax1.set_xlim(-0.85, 24.2)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xticks(x, minor=True)
    ax1.set_ylim(left_ymin, left_ymax)
    ax1.set_yticks(left_yticks)
    ax1.set_xlabel("Time (h)", fontsize=17)
    ax1.set_ylabel("Active power (MW)", fontsize=17)
    ax1.tick_params(axis="both", labelsize=13)
    ax1.grid(axis="y", which="major", linewidth=0.7, color="#b0b0b0", alpha=0.45, zorder=0)
    ax1.grid(axis="x", which="major", linewidth=0.7, color="#b0b0b0", alpha=0.35, zorder=0)
    ax1.grid(axis="x", which="minor", linewidth=0.45, color="#b0b0b0", alpha=0.22, zorder=0)
    ax1.tick_params(axis="x", which="minor", length=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    ax2 = ax1.twinx()
    (soc_line,) = ax2.plot(
        soc_x,
        soc_percent_plot,
        color=SCHEDULE_COLORS["SOC"],
        marker="D",
        markersize=4.8,
        linewidth=1.4,
        label="SOC",
        zorder=5,
    )
    right_ymin, right_ymax = secondary_ylim_align_zero(
        left_ymin=left_ymin,
        left_ymax=left_ymax,
        right_ymax=105.0,
        zero_value=0.0,
    )
    # Convert the former 0-100 usable-range axis to physical SOC while
    # preserving the trajectory positions and the existing power-axis layout.
    ax2.set_ylim(5.0 + 0.9 * right_ymin, 5.0 + 0.9 * right_ymax)
    ax2.set_yticks([5, 25, 50, 75, 95])
    ax2.set_ylabel("SOC (%)", fontsize=17)
    ax2.tick_params(axis="y", labelsize=13)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.1)
        spine.set_color("black")

    handles = [
        load_line,
        soc_line,
        bar_handles["DG"],
        bar_handles["PV"],
        bar_handles["ESS"],
        bar_handles["Grid"],
        bar_handles["WT"],
        loss_handle,
    ]
    labels = ["Load", "SOC", "DG", "PV", "ESS", "Grid", "WT", "Loss"]
    legend = ax1.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.145),
        bbox_transform=ax1.transAxes,
        ncol=8,
        fontsize=11,
        frameon=True,
        edgecolor="black",
        facecolor="white",
        framealpha=1.0,
        columnspacing=0.8,
        handlelength=1.6,
        handletextpad=0.35,
        borderpad=0.35,
        borderaxespad=0.0,
    )
    legend.get_frame().set_linewidth(1.0)
    legend.set_zorder(50)

    panel_label = method_label_for_schedule(method)
    if panel_label:
        fig.text(
            0.5,
            0.05,
            panel_label,
            ha="center",
            va="center",
            fontsize=17,
        )
    fig.subplots_adjust(left=0.085, right=0.915, top=0.945, bottom=0.215)
    savefig(fig, output_name, figure_dir, image_dir)
    plt.close(fig)


def plot_dispatch_schedules(
    dispatch_input: str,
    figure_dir: Path,
    image_dir: Optional[Path],
) -> None:
    path = resolve_path(dispatch_input)
    if not path.exists():
        print("[SKIP] dispatch trajectory raw is not ready yet:", path)
        return
    df = pd.read_csv(path)
    df = completed_rows(df)
    df = df[df["method"].isin(DISPATCH_METHODS)].copy()
    if df.empty:
        print("[SKIP] no completed 119 dispatch trajectory rows found.")
        return

    series_by_method: dict[str, dict[str, object]] = {}
    for method in DISPATCH_METHODS:
        method_rows = df[df["method"].eq(method)]
        if method_rows.empty:
            print("[MISS] dispatch trajectory for method:", method)
            continue
        series_by_method[method] = dispatch_series_from_row(method_rows.iloc[0])

    if not series_by_method:
        print("[SKIP] no requested 119 dispatch trajectory methods found.")
        return

    left_ylim, left_yticks = common_dispatch_axis(
        [series_by_method[method] for method in DISPATCH_METHODS if method in series_by_method]
    )

    for method in DISPATCH_METHODS:
        if method not in series_by_method:
            continue
        plot_single_dispatch_schedule(
            series_by_method[method],
            left_ylim=left_ylim,
            left_yticks=left_yticks,
            figure_dir=figure_dir,
            image_dir=image_dir,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate the 119-bus paper figures and scalability tables."
    )
    parser.add_argument(
        "--dispatch-input",
        default="formal/step13_119_dispatch_trajectory_raw.csv",
        help="Raw CSV generated by comparison/run_119_dispatch_trajectory.py.",
    )
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--table-dir", default=str(DEFAULT_TABLE_DIR))
    parser.add_argument(
        "--scalability-input",
        default="formal/step10_119_main_table_inline20.csv",
    )
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--no-image-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    image_dir = None if args.no_image_copy else Path(args.image_dir)

    print("119 dispatch figure dir:", figure_dir)
    print("119 image dir:", image_dir if image_dir is not None else "disabled")
    print("119 table dir:", table_dir)
    print("dispatch_input:", args.dispatch_input)
    if args.dry_run:
        return

    setup_matplotlib()
    plot_dispatch_schedules(args.dispatch_input, figure_dir, image_dir)
    write_scalability_tables(args.scalability_input, table_dir)


if __name__ == "__main__":
    main()
