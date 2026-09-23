from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from matplotlib.ticker import LogFormatterMathtext, LogLocator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_v2.config import RESULT_DIR              


FORMAL_DIR = RESULT_DIR / "formal"
DEFAULT_FIG_DIR = RESULT_DIR / "paper_figures" / "33"
DEFAULT_TABLE_DIR = RESULT_DIR / "paper_tables" / "33"
DEFAULT_IMAGE_DIR = PROJECT_ROOT / "images"
STATIC_IMAGE_DIR = PROJECT_ROOT / "paper_assets" / "static_images"

STATIC_IMAGE_NAMES = [
    "33bus.pdf",
    "119bus.pdf",
    "training_20k.pdf",
]

METHOD_ORDER_MAIN = [
    "Perfect-Global",
    "Proposed-H3",
    "MPC-3",
    "MPC-5",
    "MPC-7",
    "StdNN-H3-MIP",
    "SAC",
    "SAC-ICNN",
    "SACLag",
]

COST_COLUMN_MAP = {
    "Perfect-Global": "opt",
    "Proposed-H3": "proposed",
    "MPC-3": "mpc3",
    "MPC-5": "mpc5",
    "MPC-7": "mpc7",
    "StdNN-H3-MIP": "stdnn",
    "SAC": "sac",
    "SAC-ICNN": "sacicnn",
    "SACLag": "saclag",
}

TIME_COLUMN_MAP = {
    "Proposed-H3": "proposed",
    "MPC-3": "mpc3",
    "MPC-5": "mpc5",
    "MPC-7": "mpc7",
    "StdNN-H3-MIP": "stdnn",
}

TIME33_BOX_COLORS = {
    "Proposed-H3": "#4C78A8",
    "MPC-3": "#F58518",
    "MPC-5": "#54A24B",
    "MPC-7": "#B279A2",
    "StdNN-H3-MIP": "#E45756",
}

TIME33_BOX_HATCHES = {
    "Proposed-H3": "",
    "MPC-3": "//",
    "MPC-5": "\\\\",
    "MPC-7": "xx",
    "StdNN-H3-MIP": "..",
}

NN_EMBEDDING_ORDER = ["MPC-3", "StdNN-H3-MIP", "Proposed-H3"]
MODEL_COMPLEXITY_ORDER = ["MPC-3", "MPC-5", "MPC-7", "StdNN-H3-MIP", "Proposed-H3"]
STDNN_ONLY_RAW = "step6_33_stdnn_only_raw.csv"
STDNN_ONLY_SUMMARY = "step6_33_stdnn_only_summary.csv"
SACLAG_STOCHASTIC_RAW = "step7_33_saclag_stochastic_seed0_test_raw.csv"
MODEL_COMPLEXITY_SUMMARY = "step6_33_model_complexity_summary.csv"

FORECAST_CASE_ORDER = ["id_reliable", "high_error", "biased_ood"]
FORECAST_CASE_LABELS = {
    "id_reliable": "Reliable",
    "high_error": "High-error",
    "biased_ood": "Biased OOD",
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


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(relative_name: str, *, required: bool = True, **kwargs: object) -> Optional[pd.DataFrame]:
    path = FORMAL_DIR / relative_name
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        print("[MISS]", path)
        return None
    return pd.read_csv(path, **kwargs)


def bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def missing_value(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def stdnn_raw_candidates() -> List[tuple[pd.DataFrame, int]]:
    candidates: List[tuple[pd.DataFrame, int]] = []
    for relative_name, source_rank in [
        (STDNN_ONLY_RAW, 1),
        ("step6_33_nn_embedding_raw.csv", 2),
    ]:
        df = read_csv(relative_name, required=False)
        if df is None:
            continue
        current = df.copy()
        if "completed" in current.columns:
            current = current[bool_series(current["completed"])]
        current = current[current["method"].astype(str).eq("StdNN-H3-MIP")].copy()
        current["_source_rank"] = source_rank
        candidates.append((current, source_rank))
    return candidates


def policy_raw_candidates() -> List[tuple[pd.DataFrame, int]]:
    candidates: List[tuple[pd.DataFrame, int]] = []
    for relative_name, source_rank, method_filter in [
        ("step7_33_policy_baselines_raw.csv", 0, None),
        (SACLAG_STOCHASTIC_RAW, -1, "SACLag"),
    ]:
        df = read_csv(relative_name, required=relative_name == "step7_33_policy_baselines_raw.csv")
        if df is None:
            continue
        current = df.copy()
        if "completed" in current.columns:
            current = current[bool_series(current["completed"])]
        if method_filter is not None:
            current = current[current["method"].astype(str).eq(method_filter)].copy()
        current["_source_rank"] = source_rank
        candidates.append((current, source_rank))
    return candidates


def first_summary_row(df: Optional[pd.DataFrame], method: str) -> Optional[pd.Series]:
    if df is None or "method" not in df.columns:
        return None
    rows = df[df["method"].astype(str).eq(method)].copy()
    if rows.empty:
        return None
    if "n_runs" in rows.columns:
        rows["_n_runs_sort"] = numeric(rows["n_runs"]).fillna(-1)
        rows = rows.sort_values("_n_runs_sort", ascending=False)
    return rows.iloc[0]


def collect_33_nn_embedding_summary() -> pd.DataFrame:
    main_summary = read_csv("step4_33_main_model_based_summary.csv", required=False)
    nn_summary = read_csv("step6_33_nn_embedding_summary.csv", required=False)
    stdnn_summary = read_csv(STDNN_ONLY_SUMMARY, required=False)

    rows = []
    model_fields = [
        "mean_num_vars",
        "mean_num_bin_vars",
        "mean_num_constrs",
        "mean_num_qconstrs",
        "mean_num_genconstrs",
        "mean_model_runtime_s",
        "mean_mip_gap",
        "max_mip_gap",
    ]
    for method in NN_EMBEDDING_ORDER:
        if method == "StdNN-H3-MIP":
            perf_sources = [stdnn_summary, nn_summary, main_summary]
            model_sources = [stdnn_summary, nn_summary, main_summary]
        else:
            perf_sources = [main_summary, nn_summary]
            model_sources = [nn_summary, main_summary]

        perf_row = next(
            (row for row in (first_summary_row(df, method) for df in perf_sources) if row is not None),
            None,
        )
        if perf_row is None:
            continue

        merged = perf_row.to_dict()
        model_row = next(
            (row for row in (first_summary_row(df, method) for df in model_sources) if row is not None),
            None,
        )
        if model_row is not None:
            for field in model_fields:
                if field in model_row and missing_value(merged.get(field)):
                    merged[field] = model_row[field]
        rows.append(merged)

    if not rows:
        raise FileNotFoundError("No 33-bus NN embedding summary rows were found.")

    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], NN_EMBEDDING_ORDER, ordered=True)
    return df.sort_values("method")


def collect_33_model_complexity_summary() -> pd.DataFrame:
    probe = read_csv(MODEL_COMPLEXITY_SUMMARY, required=False)
    if probe is not None and not probe.empty:
        out = probe[probe["method"].astype(str).isin(MODEL_COMPLEXITY_ORDER)].copy()
        out["method"] = pd.Categorical(out["method"], MODEL_COMPLEXITY_ORDER, ordered=True)
        return out.sort_values("method")

    main_summary = read_csv("step4_33_main_model_based_summary.csv", required=False)
    nn_summary = read_csv("step6_33_nn_embedding_summary.csv", required=False)
    stdnn_summary = read_csv(STDNN_ONLY_SUMMARY, required=False)
    sources = [df for df in [main_summary, stdnn_summary, nn_summary] if df is not None]
    rows = []
    for method in MODEL_COMPLEXITY_ORDER:
        selected: Optional[pd.Series] = None
        for source in sources:
            if "method" not in source.columns:
                continue
            candidates = source[source["method"].astype(str).eq(method)].copy()
            if candidates.empty:
                continue
            if "mean_num_vars" in candidates.columns:
                candidates["_has_size"] = numeric(candidates["mean_num_vars"]).notna()
                sort_cols = ["_has_size"]
            else:
                sort_cols = []
            if "n_runs" in candidates.columns:
                candidates["_n_runs_sort"] = numeric(candidates["n_runs"]).fillna(-1)
                sort_cols.append("_n_runs_sort")
            if sort_cols:
                candidates = candidates.sort_values(sort_cols, ascending=False)
            selected = candidates.iloc[0]
            break
        if selected is None:
            continue

        total_vars = pd.to_numeric(pd.Series([selected.get("mean_num_vars")]), errors="coerce").iloc[0]
        bin_vars = pd.to_numeric(pd.Series([selected.get("mean_num_bin_vars")]), errors="coerce").iloc[0]
        linear_constrs = pd.to_numeric(pd.Series([selected.get("mean_num_constrs")]), errors="coerce").iloc[0]
        q_constrs = pd.to_numeric(pd.Series([selected.get("mean_num_qconstrs", 0)]), errors="coerce").fillna(0).iloc[0]
        gen_constrs = pd.to_numeric(pd.Series([selected.get("mean_num_genconstrs", 0)]), errors="coerce").fillna(0).iloc[0]
        rows.append(
            {
                "method": method,
                "cont_vars": total_vars - bin_vars,
                "bin_vars": bin_vars,
                "total_vars": total_vars,
                "linear_constrs": linear_constrs,
                "quadratic_constrs": q_constrs,
                "general_constrs": gen_constrs,
                "total_constrs": linear_constrs + q_constrs + gen_constrs,
            }
        )

    if not rows:
        raise FileNotFoundError(
            "No 33-bus model-complexity rows were found. Run "
            "`python -m experiments_v2.summary.probe_33_model_complexity` first."
        )

    out = pd.DataFrame(rows)
    out["method"] = pd.Categorical(out["method"], MODEL_COMPLEXITY_ORDER, ordered=True)
    return out.sort_values("method")


def parse_json_list(value: object, expected_len: int = 48) -> np.ndarray:
    if value is None or pd.isna(value):
        return np.full(expected_len, np.nan)

    try:
        data = json.loads(str(value))
    except Exception:
        return np.full(expected_len, np.nan)

    try:
        arr = np.asarray(data, dtype=float).reshape(-1)
    except Exception:
        return np.full(expected_len, np.nan)
    if arr.size < expected_len:
        arr = np.pad(arr, (0, expected_len - arr.size), constant_values=np.nan)
    return arr[:expected_len]


def ci95(values: Iterable[object]) -> float:
    v = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(v) < 2:
        return math.nan
    return 1.96 * float(v.std(ddof=1)) / math.sqrt(len(v))


def savefig(
    fig: plt.Figure,
    name: str,
    figure_dir: Path,
    image_dir: Optional[Path],
    *,
    bbox_inches: Optional[str] = "tight",
) -> None:
    out_pdf = figure_dir / name
    fig.savefig(out_pdf, bbox_inches=bbox_inches)
    print("Saved:", out_pdf)

    if image_dir is not None:
        image_dir.mkdir(parents=True, exist_ok=True)
        latex_out = image_dir / name
        fig.savefig(latex_out, bbox_inches=bbox_inches)
        print("Saved:", latex_out)


def copy_static_images(image_dir: Optional[Path]) -> None:
    if image_dir is None:
        return
    if not STATIC_IMAGE_DIR.exists():
        print("[SKIP] static image directory is not ready:", STATIC_IMAGE_DIR)
        return

    image_dir.mkdir(parents=True, exist_ok=True)
    for name in STATIC_IMAGE_NAMES:
        src = STATIC_IMAGE_DIR / name
        dst = image_dir / name
        if not src.exists():
            print("[MISS]", src)
            continue
        shutil.copy2(src, dst)
        print("Copied:", dst)


def collect_33_cost_raw() -> pd.DataFrame:
    sources = [
        (read_csv("step7b_33_perfect_global_raw.csv"), 0),
        (read_csv("step4_33_main_model_based_raw.csv"), 0),
    ]
    frames: List[pd.DataFrame] = []
    for df, source_rank in sources:
        assert df is not None
        current = df.copy()
        if "completed" in current.columns:
            current = current[bool_series(current["completed"])]
        current["_source_rank"] = source_rank
        frames.append(current)

    for policy, _source_rank in policy_raw_candidates():
        frames.append(policy)

    for stdnn, _source_rank in stdnn_raw_candidates():
        frames.append(stdnn)

    raw = pd.concat(frames, ignore_index=True, sort=False)
    raw["scenario_id"] = numeric(raw["scenario_id"])
    raw = raw[
        (raw["system"].astype(str) == "33")
        & (raw["actual_case"].astype(str) == "id_actual")
        & raw["method"].isin(METHOD_ORDER_MAIN)
        & (raw["scenario_id"] >= 80)
        & (raw["scenario_id"] <= 99)
    ].copy()
    if "forecast_case" in raw.columns:
        forecast_text = raw["forecast_case"].fillna("").astype(str)
        raw = raw[
            raw["method"].astype(str).eq("Perfect-Global")
            | forecast_text.isin(["", "id_reliable", "not_applicable", "perfect_full_horizon"])
        ].copy()
    raw = raw.sort_values(["method", "scenario_id", "_source_rank"])
    raw = raw.drop_duplicates(subset=["method", "scenario_id"], keep="first")
    raw["scenario_index"] = raw["scenario_id"].astype(int) - 79
    raw["cost"] = numeric(raw["cost"])
    return raw.drop(columns=["_source_rank"], errors="ignore")


def export_legacy_cost_xlsx(figure_dir: Path) -> Path:
    raw = collect_33_cost_raw()
    pivot = raw.pivot_table(
        index="scenario_index",
        columns="method",
        values="cost",
        aggfunc="first",
    ).sort_index()

    out = pd.DataFrame({"scenario_index": pivot.index})
    for method, column in COST_COLUMN_MAP.items():
        out[column] = pivot[method].values if method in pivot.columns else np.nan

    path = figure_dir / "cost33_v2.xlsx"
    out.to_excel(path, index=False)
    print("Saved:", path)
    return path


def plot_cost33(figure_dir: Path, image_dir: Optional[Path]) -> None:
    xlsx = export_legacy_cost_xlsx(figure_dir)
    df = pd.read_excel(xlsx)
    x = numeric(df["scenario_index"]).to_numpy()

    fig, ax = plt.subplots(figsize=(7.2, 3.35))
    plot_specs = [
        ("opt", "Perfect-Global", "o", "-", 1.2),
        ("proposed", "Proposed-H3", "s", "-", 2.0),
        ("mpc3", "MPC-3", "^", "--", 1.2),
        ("mpc5", "MPC-5", ">", (0, (5, 2)), 1.2),
        ("mpc7", "MPC-7", "<", ":", 1.5),
        ("stdnn", "StdNN-H3-MIP", "X", (0, (2, 1)), 1.2),
        ("sac", "SAC", "D", (0, (3, 1, 1, 1)), 1.2),
        ("sacicnn", "SAC-ICNN", "v", "-.", 1.2),
        ("saclag", "SACLag", "*", (0, (1, 1)), 1.3),
    ]

    for column, label, marker, linestyle, line_width in plot_specs:
        if column not in df.columns:
            continue
        values = numeric(df[column])
        if values.notna().any():
            ax.plot(
                x,
                values.to_numpy(),
                label=label,
                linewidth=line_width,
                linestyle=linestyle,
                marker=marker,
                markersize=4,
            )

    ax.set_xlabel("Test scenario index", fontsize=12)
    ax.set_ylabel("Total cost ($)", fontsize=12)
    ax.set_xlim(0.5, len(x) + 0.5)
    ax.set_xticks(np.arange(1, len(x) + 1, 1))
    ax.grid(axis="both", linestyle="--", alpha=0.55)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.30),
        ncol=5,
        frameon=True,
        edgecolor="black",
        fontsize=8.5,
        handlelength=1.8,
        columnspacing=0.75,
        handletextpad=0.35,
    )
    fig.tight_layout()
    savefig(fig, "cost33.pdf", figure_dir, image_dir)
    plt.close(fig)


def export_legacy_time_xlsx(figure_dir: Path) -> Path:
    raw_parts: List[pd.DataFrame] = []

    main_raw = read_csv("step4_33_main_model_based_raw.csv")
    assert main_raw is not None
    main_raw = main_raw.copy()
    main_raw["_source_rank"] = 0
    raw_parts.append(main_raw)

    for stdnn_raw, _source_rank in stdnn_raw_candidates():
        raw_parts.append(stdnn_raw)

    raw = pd.concat(raw_parts, ignore_index=True, sort=False)
    raw = raw[bool_series(raw["completed"])].copy()
    raw["scenario_id"] = numeric(raw["scenario_id"])
    raw = raw[
        (raw["system"].astype(str) == "33")
        & (raw["actual_case"].astype(str) == "id_actual")
        & (raw["forecast_case"].astype(str) == "id_reliable")
        & raw["method"].isin(TIME_COLUMN_MAP)
        & (raw["scenario_id"] >= 80)
        & (raw["scenario_id"] <= 99)
    ].copy()
    raw = raw.sort_values(["method", "scenario_id", "_source_rank"])
    raw = raw.drop_duplicates(subset=["method", "scenario_id"], keep="first")

    raw["scenario_index"] = raw["scenario_id"].astype(int) - 79
    raw["solve_time_s"] = numeric(raw["solve_time_s"])
    raw["executed_steps"] = numeric(raw["executed_steps"])
    raw["avg_step_time_s"] = numeric(raw["avg_step_time_s"])
    missing = raw["avg_step_time_s"].isna() & raw["solve_time_s"].notna() & (raw["executed_steps"] > 0)
    raw.loc[missing, "avg_step_time_s"] = raw.loc[missing, "solve_time_s"] / raw.loc[
        missing, "executed_steps"
    ]

    pivot = raw.pivot_table(
        index="scenario_index",
        columns="method",
        values="avg_step_time_s",
        aggfunc="first",
    ).sort_index()

    out = pd.DataFrame({"scenario_index": pivot.index})
    for method, column in TIME_COLUMN_MAP.items():
        out[column] = pivot[method].values if method in pivot.columns else np.nan

    path = figure_dir / "time33_v2.xlsx"
    out.to_excel(path, index=False)
    print("Saved:", path)
    return path


def plot_time33(figure_dir: Path, image_dir: Optional[Path]) -> None:
    xlsx = export_legacy_time_xlsx(figure_dir)
    df = pd.read_excel(xlsx)

    columns = ["proposed", "mpc3", "mpc5", "mpc7", "stdnn"]
    labels = ["Proposed-H3", "MPC-3", "MPC-5", "MPC-7", "StdNN-H3-MIP"]
    data = []
    for column in columns:
        values = numeric(df[column]).dropna().to_numpy()
        data.append(np.where(values <= 0, 1e-6, values))

    fig, ax = plt.subplots(figsize=(4.7, 3.15))
    boxplot_kwargs = dict(
        showfliers=True,
        whis=(5, 95),
        patch_artist=True,
        widths=0.55,
    )
    try:
        bp = ax.boxplot(data, tick_labels=labels, **boxplot_kwargs)
    except TypeError:
        bp = ax.boxplot(data, labels=labels, **boxplot_kwargs)
    for box, label in zip(bp["boxes"], labels):
        box.set_facecolor(TIME33_BOX_COLORS[label])
        box.set_alpha(0.65)
        box.set_edgecolor("black")
        box.set_linewidth(1.4)
        box.set_hatch(TIME33_BOX_HATCHES[label])
    for median in bp["medians"]:
        median.set(color="black", linewidth=1.3)
    for whisker in bp["whiskers"]:
        whisker.set(color="black", linewidth=1.1)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.1)
    for flier, label in zip(bp["fliers"], labels):
        flier.set_marker("o")
        flier.set_markersize(3.2)
        flier.set_markerfacecolor("white")
        flier.set_markeredgecolor(TIME33_BOX_COLORS[label])
        flier.set_alpha(0.9)

    ax.set_yscale("log")
    ax.set_ylabel("Average per-step computation time (s, log scale)", fontsize=10)
    ax.grid(axis="y", which="major", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=5))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10))
    ax.tick_params(axis="x", labelsize=11)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    fig.tight_layout()
    savefig(fig, "time33.pdf", figure_dir, image_dir)
    plt.close(fig)


def plot_gamma_validation(figure_dir: Path, image_dir: Optional[Path]) -> None:
    df = read_csv("step3_33_gamma_validation_summary.csv")
    assert df is not None
    df = df.copy()
    df["gamma_Q"] = numeric(df["gamma_Q"])
    df["mean_cost"] = numeric(df["mean_cost"])
    df = df.sort_values("gamma_Q")

    fig, ax = plt.subplots(figsize=(4.35, 3.15))
    ax.plot(df["gamma_Q"], df["mean_cost"], marker="o", linewidth=1.6, markersize=5)
    ax.set_xlabel(r"Scale factor $\gamma_Q$", fontsize=12)
    ax.set_ylabel("Validation cost ($)", fontsize=12)

    best = df.loc[df["mean_cost"].idxmin()]
    best_gamma = float(best["gamma_Q"])
    best_cost = float(best["mean_cost"])
    ax.scatter([best_gamma], [best_cost], s=55, zorder=5)

    y_min = float(df["mean_cost"].min())
    y_max = float(df["mean_cost"].max())
    y_range = y_max - y_min
    y_pad = 0.08 * y_range if y_range > 0 else max(1.0, abs(y_max) * 0.05)
    ax.annotate(
        rf"Selected $\gamma_Q={best_gamma:.1f}$",
        xy=(best_gamma, best_cost),
        xytext=(min(best_gamma + 0.10, 0.72), best_cost + 0.55 * y_pad),
        arrowprops=dict(arrowstyle="->", linewidth=0.8),
        fontsize=10,
    )

    ax.set_xlim(-0.03, 1.03)
    ax.set_xticks(np.arange(0.0, 1.01, 0.2))
    ax.set_xticks(np.arange(0.0, 1.01, 0.1), minor=True)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.grid(axis="both", which="major", linestyle="--", alpha=0.55)
    ax.grid(axis="x", which="minor", linestyle=":", alpha=0.30)

    fig.tight_layout()
    savefig(fig, "gamma33.pdf", figure_dir, image_dir)
    plt.close(fig)


def plot_forecast_sensitivity(figure_dir: Path, image_dir: Optional[Path]) -> None:
    pair = read_csv("step5_33_pairwise_mpc3_vs_proposed_by_case.csv")
    assert pair is not None
    pair = pair.copy()
    pair["forecast_case"] = pd.Categorical(pair["forecast_case"], FORECAST_CASE_ORDER, ordered=True)
    pair = pair.sort_values("forecast_case")
    pair["mean_cost_reduction_percent"] = numeric(pair["mean_cost_reduction_percent"])
    pair["ci95_cost_reduction_percent"] = numeric(pair["ci95_cost_reduction_percent"])

    x = np.arange(len(pair))
    y = pair["mean_cost_reduction_percent"].to_numpy()
    e = pair["ci95_cost_reduction_percent"].fillna(0.0).to_numpy()
    labels = [FORECAST_CASE_LABELS.get(str(case), str(case)) for case in pair["forecast_case"]]

    fig, ax = plt.subplots(figsize=(4.35, 3.1))
    ax.bar(
        x,
        y,
        yerr=e,
        capsize=4,
        width=0.62,
        edgecolor="black",
        linewidth=0.7,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cost reduction vs. MPC-3 (%)", fontsize=12)
    ax.set_xlabel("Forecast case", fontsize=12)
    ax.grid(axis="y", linestyle="--", alpha=0.55)

                                                                       
                                                  
    y_top = float(np.nanmax(y + e)) if len(y) else 1.0
    ax.set_ylim(0.0, y_top * 1.12)

    fig.tight_layout()
    savefig(fig, "forecast_sensitivity33.pdf", figure_dir, image_dir)
    plt.close(fig)


def plot_nn_embedding(figure_dir: Path, image_dir: Optional[Path]) -> None:
    df = collect_33_nn_embedding_summary()

    for column in ["mean_solve_time_s", "ci95_solve_time_s", "mean_num_bin_vars"]:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = numeric(df[column])

    x = np.arange(len(df))
    width = 0.38
    fig, ax1 = plt.subplots(figsize=(4.8, 3.15))

    ax1.bar(
        x - width / 2,
        df["mean_solve_time_s"],
        width=width,
        yerr=df["ci95_solve_time_s"].fillna(0.0),
        capsize=3,
        label="Runtime",
    )
    ax1.set_ylabel("Daily runtime (s)", fontsize=12)
    ax1.set_yscale("log")
    ax1.grid(axis="y", linestyle="--", alpha=0.55)

    ax2 = ax1.twinx()
    ax2.bar(
        x + width / 2,
        df["mean_num_bin_vars"].fillna(0.0),
        width=width,
        label="Binary variables",
        alpha=0.55,
    )
    ax2.set_ylabel("Binary variables", fontsize=12)

    ax1.set_xticks(x)
    ax1.set_xticklabels(df["method"].astype(str), rotation=15)
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        handles1 + handles2,
        labels1 + labels2,
        loc="upper left",
        fontsize=9,
        frameon=True,
        edgecolor="black",
    )

    fig.tight_layout()
    savefig(fig, "nn_embedding33.pdf", figure_dir, image_dir)
    plt.close(fig)


SCHEDULE_METHOD_TO_FILE = {
    "Proposed-H3": "scheduling_results_P_pro.pdf",
    "SAC": "scheduling_results_P_sac.pdf",
    "MPC-3": "scheduling_results_P_mpc.pdf",
}

OLD_SCHEDULE_COLORS = {
    "Load": "#513743",
    "SOC": "#666600",
    "DG": "#cccc66",
    "PV": "#ffcc33",
    "ESS": "#b39cd0",
    "Grid": "#cc9966",
    "WT": "#669999",
    "Loss": "#669933",
}


def method_label_for_schedule(method: str) -> str:
    if method == "Proposed-H3":
        return "(a)"
    if method == "SAC":
        return "(b)"
    if method == "MPC-3":
        return "(c)"
    return ""


def _bar_values(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)


def secondary_ylim_align_zero(
    *,
    left_ymin: float,
    left_ymax: float,
    right_ymax: float,
    zero_value: float = 0.0,
) -> tuple[float, float]:
                                                                                   
    if not (left_ymin < zero_value < left_ymax):
        raise ValueError("zero_value must lie inside the left y-axis range.")

    frac = (zero_value - left_ymin) / (left_ymax - left_ymin)
    right_ymin = zero_value - frac / (1.0 - frac) * (right_ymax - zero_value)
    return right_ymin, right_ymax


def plot_single_dispatch_schedule(
    row: pd.Series,
    *,
    figure_dir: Path,
    image_dir: Optional[Path],
) -> None:
    method = str(row["method"])
    output_name = SCHEDULE_METHOD_TO_FILE.get(method)
    if output_name is None:
        return

    n_steps = 48
    x = np.arange(n_steps) / 2.0

    load = parse_json_list(row.get("step_load_total_json"), n_steps)
    soc_mwh = parse_json_list(row.get("step_SOC_BSS_9_json"), n_steps)
    p_dg = parse_json_list(row.get("step_P_DG_5_json"), n_steps)
    p_bss = parse_json_list(row.get("step_P_BSS_json"), n_steps)
    p_grid = parse_json_list(row.get("step_P_grid_json"), n_steps)
    loss = parse_json_list(row.get("step_loss_json"), n_steps)
    pv = parse_json_list(row.get("step_pv_11_json"), n_steps)
    wt = parse_json_list(row.get("step_wind_26_json"), n_steps)

    soc_min = 0.125
    soc_max = 2.375
    soc_percent = 100.0 * (soc_mwh - soc_min) / (soc_max - soc_min)
    soc_percent = np.clip(soc_percent, 0.0, 100.0)
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
        heights = _bar_values(values)
        handle = ax1.bar(
            x,
            heights,
            width=width,
            bottom=bottom_pos,
            color=OLD_SCHEDULE_COLORS[label],
            edgecolor="none",
            linewidth=0.0,
            alpha=1.0,
            label=label,
            zorder=2,
        )
        bottom_pos = bottom_pos + heights
        bar_handles[label] = handle

    bottom_neg = np.zeros(n_steps)
    grid_neg_heights = _bar_values(grid_neg)
    grid_neg_handle = ax1.bar(
        x,
        grid_neg_heights,
        width=width,
        bottom=bottom_neg,
        color=OLD_SCHEDULE_COLORS["Grid"],
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
        label="Grid",
        zorder=2,
    )
    bottom_neg = bottom_neg + grid_neg_heights
    ess_neg_heights = _bar_values(ess_neg)
    ess_neg_handle = ax1.bar(
        x,
        ess_neg_heights,
        width=width,
        bottom=bottom_neg,
        color=OLD_SCHEDULE_COLORS["ESS"],
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
        label="ESS",
        zorder=2,
    )
    bottom_neg = bottom_neg + ess_neg_heights
    loss_handle = ax1.bar(
        x,
        _bar_values(loss_neg),
        width=width,
        bottom=bottom_neg,
        color=OLD_SCHEDULE_COLORS["Loss"],
        edgecolor="none",
        linewidth=0.0,
        alpha=1.0,
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
        color=OLD_SCHEDULE_COLORS["Load"],
        marker="*",
        markersize=7.0,
        linewidth=1.7,
        label="Load",
        zorder=4,
    )

    ax1.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    left_ymin = -3.0
    left_ymax = 5.5
    ax1.set_xlim(-0.85, 24.2)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_ylim(left_ymin, left_ymax)
    ax1.set_yticks([-3, -2, -1, 0, 1, 2, 3, 4, 5])
    ax1.set_xlabel("Time (h)", fontsize=17)
    ax1.set_ylabel("Active power (MW)", fontsize=17)
    ax1.tick_params(axis="both", labelsize=13)
    ax1.set_xticks(np.arange(0, 25, 2))
    ax1.set_xticks(x, minor=True)
    ax1.grid(
        axis="y",
        which="major",
        linestyle="-",
        linewidth=0.7,
        color="#b0b0b0",
        alpha=0.45,
        zorder=0,
    )
    ax1.grid(
        axis="x",
        which="major",
        linestyle="-",
        linewidth=0.7,
        color="#b0b0b0",
        alpha=0.45,
        zorder=0,
    )
    ax1.grid(
        axis="x",
        which="minor",
        linestyle="-",
        linewidth=0.45,
        color="#b0b0b0",
        alpha=0.25,
        zorder=0,
    )
    ax1.tick_params(axis="x", which="minor", length=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color("black")

    ax2 = ax1.twinx()
    (soc_line,) = ax2.plot(
        soc_x,
        soc_percent_plot,
        color=OLD_SCHEDULE_COLORS["SOC"],
        marker="D",
        markersize=5.2,
        linewidth=1.5,
        label="SOC",
        zorder=5,
    )
    right_ymin, right_ymax = secondary_ylim_align_zero(
        left_ymin=left_ymin,
        left_ymax=left_ymax,
        right_ymax=105.0,
        zero_value=0.0,
    )
    ax2.set_ylim(right_ymin, right_ymax)
    ax2.set_yticks([0, 25, 50, 75, 100])
    ax2.set_ylabel("SOC (%)", fontsize=17)
    ax2.tick_params(axis="y", labelsize=13)
    for spine in ax2.spines.values():
        spine.set_linewidth(1.2)
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
    )
    legend.get_frame().set_linewidth(1.0)
    legend.set_zorder(50)

    fig.subplots_adjust(
        left=0.085,
        right=0.915,
        top=0.945,
        bottom=0.215,
    )

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

    savefig(fig, output_name, figure_dir, image_dir, bbox_inches=None)
    plt.close(fig)


def plot_dispatch_schedules(figure_dir: Path, image_dir: Optional[Path]) -> None:
    df = read_csv("step9_33_dispatch_trajectory_raw.csv", required=False)
    if df is None:
        print("[SKIP] dispatch trajectory raw is not ready yet.")
        return

    if "completed" in df.columns:
        df = df[bool_series(df["completed"])].copy()

    wanted = ["Proposed-H3", "SAC", "MPC-3"]
    df = df[df["method"].isin(wanted)].copy()
    if df.empty:
        print("[SKIP] no completed dispatch trajectory rows found.")
        return

    for method in wanted:
        method_rows = df[df["method"] == method]
        if method_rows.empty:
            print("[MISS] dispatch trajectory for method:", method)
            continue
        plot_single_dispatch_schedule(
            method_rows.iloc[0],
            figure_dir=figure_dir,
            image_dir=image_dir,
        )


def fmt_mean_ci(mean_value: object, ci_value: object, digits: int = 2) -> str:
    mean_num = pd.to_numeric(pd.Series([mean_value]), errors="coerce").iloc[0]
    ci_num = pd.to_numeric(pd.Series([ci_value]), errors="coerce").iloc[0]
    if pd.isna(mean_num):
        return "--"
    if pd.isna(ci_num):
        return f"{float(mean_num):.{digits}f}"
    return f"{float(mean_num):.{digits}f} $\\pm$ {float(ci_num):.{digits}f}"


def fmt_violation(row: pd.Series) -> str:
    method = str(row.get("method", ""))
    mean_num = pd.to_numeric(
        pd.Series([row.get("mean_voltage_violation", np.nan)]),
        errors="coerce",
    ).iloc[0]
    ci_num = pd.to_numeric(
        pd.Series([row.get("ci95_voltage_violation", np.nan)]),
        errors="coerce",
    ).iloc[0]

    if pd.isna(mean_num):
        if method in {"SAC", "SAC-ICNN", "SACLag"}:
            return "--"
        mean_num = pd.to_numeric(
            pd.Series([row.get("mean_cost_constraint", np.nan)]),
            errors="coerce",
        ).iloc[0]
        ci_num = np.nan

    if pd.isna(mean_num) or abs(float(mean_num)) <= 5e-10:
        return "0"
    if pd.isna(ci_num):
        return f"{float(mean_num):.4f}"
    return f"{float(mean_num):.4f} $\\pm$ {float(ci_num):.4f}"


def tex_table(df: pd.DataFrame, output_path: Path, column_format: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    headers = [str(column) for column in df.columns]
    lines = [
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in df.to_numpy():
        lines.append(" & ".join(str(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    tex = "\n".join(lines)
    output_path.write_text(tex, encoding="utf-8")
    print("Saved:", output_path)


def make_main_table_tex(table_dir: Path) -> None:
    table = read_csv("step8_33_main_table_with_perfect_gap.csv")
    assert table is not None
    keep = [
        "Perfect-Global",
        "Proposed-H3",
        "MPC-3",
        "MPC-5",
        "MPC-7",
        "StdNN-H3-MIP",
        "SAC",
        "SAC-ICNN",
        "SACLag",
    ]
    table = table[table["method"].isin(keep)].copy()
    table["method"] = pd.Categorical(table["method"], keep, ordered=True)
    table = table.sort_values("method")

    raw_parts: List[pd.DataFrame] = []
    for raw_name, source_rank in [
        ("step4_33_main_model_based_raw.csv", 0),
    ]:
        raw_df = read_csv(raw_name)
        assert raw_df is not None
        raw_df = raw_df.copy()
        raw_df["_source_rank"] = source_rank
        raw_parts.append(raw_df)

    for policy_raw, _source_rank in policy_raw_candidates():
        raw_parts.append(policy_raw)

    for stdnn_raw, _source_rank in stdnn_raw_candidates():
        raw_parts.append(stdnn_raw)

    time_raw = pd.concat(raw_parts, ignore_index=True, sort=False)
    time_raw = time_raw[bool_series(time_raw["completed"])].copy()
    time_raw["scenario_id"] = numeric(time_raw["scenario_id"])
    time_raw = time_raw[
        (time_raw["system"].astype(str) == "33")
        & (time_raw["actual_case"].astype(str) == "id_actual")
        & time_raw["method"].isin(keep)
        & (time_raw["scenario_id"] >= 80)
        & (time_raw["scenario_id"] <= 99)
    ].copy()
    forecast_text = time_raw["forecast_case"].fillna("").astype(str)
    time_raw = time_raw[
        forecast_text.isin(["", "id_reliable", "not_applicable", "perfect_full_horizon"])
    ].copy()
    time_raw = time_raw.sort_values(["method", "scenario_id", "_source_rank"])
    time_raw = time_raw.drop_duplicates(subset=["method", "scenario_id"], keep="first")
    time_raw["avg_step_time_s"] = numeric(time_raw["avg_step_time_s"])
    time_ci = (
        time_raw.groupby("method")["avg_step_time_s"]
        .agg(mean_avg_step_time_s_raw="mean", ci95_avg_step_time_s=ci95)
        .reset_index()
    )

    table = table.merge(time_ci, on="method", how="left")
    rows = []
    for _, row in table.iterrows():
        method = str(row["method"])
        if method == "Perfect-Global":
            gap = "--"
            time_text = "--"
        else:
            gap = fmt_mean_ci(
                row.get("mean_gap_to_perfect_percent", np.nan),
                row.get("ci95_gap_to_perfect_percent", np.nan),
                2,
            )
            time_text = fmt_mean_ci(
                row.get("mean_avg_step_time_s_raw", row.get("mean_avg_step_time_s", np.nan)),
                row.get("ci95_avg_step_time_s", np.nan),
                3,
            )

        rows.append(
            {
                "Method": method,
                "Cost ($)": fmt_mean_ci(row.get("mean_cost"), row.get("ci95_cost"), 2),
                "Gap (%)": gap,
                "Time (s)": time_text,
                "Completed": f"{int(row['completed_count'])}/{int(row['n_runs'])}",
                "Vio. (p.u.)": fmt_violation(row),
            }
        )

    tex_table(pd.DataFrame(rows), table_dir / "table_33_main_compact.tex", "lccccc")


def make_gamma_table_tex(table_dir: Path) -> None:
    df = read_csv("step3_33_gamma_validation_summary.csv")
    assert df is not None
    df = df.copy()
    df["gamma_Q"] = numeric(df["gamma_Q"])
    cost_col = "mean_completed_cost" if "mean_completed_cost" in df else "mean_cost"
    time_col = (
        "mean_completed_avg_step_time_s"
        if "mean_completed_avg_step_time_s" in df
        else "mean_avg_step_time_s"
        if "mean_avg_step_time_s" in df
        else ""
    )
    vio_col = (
        "mean_completed_cum_voltage_violation"
        if "mean_completed_cum_voltage_violation" in df
        else "mean_cum_voltage_violation"
        if "mean_cum_voltage_violation" in df
        else ""
    )
    df[cost_col] = numeric(df[cost_col])
    if time_col:
        df[time_col] = numeric(df[time_col])
    else:
        df["mean_time_s"] = numeric(df["mean_time_s"])
    if vio_col:
        df[vio_col] = numeric(df[vio_col])
    df = df.sort_values("gamma_Q")

    def gamma_label(value: object) -> str:
        gamma = float(value)
        if abs(gamma) <= 1e-12:
            return "0 (MPC-3)"
        if abs(gamma - round(gamma)) <= 1e-12:
            return f"{gamma:.0f}"
        return f"{gamma:.1f}"

    def row_values(rows: pd.DataFrame, *, kind: str) -> List[str]:
        values = []
        for _, item in rows.iterrows():
            if kind == "cost":
                values.append(f"{float(item[cost_col]):.2f}")
            elif kind == "time":
                if time_col:
                    value = float(item[time_col])
                else:
                    value = float(item["mean_time_s"]) / 48.0
                values.append(f"{value:.3f}")
            elif kind == "vio":
                value = float(item[vio_col]) if vio_col else 0.0
                values.append("0" if abs(value) <= 5e-7 else f"{value:.4f}")
        return values

    perfect_summary = read_csv("step3_33_perfect_global_validation_summary.csv", required=False)
    perfect_cost = "--"
    perfect_vio = "--"
    if perfect_summary is not None and not perfect_summary.empty:
        perfect_rows = perfect_summary[
            perfect_summary["method"].astype(str).eq("Perfect-Global")
        ].copy()
        if not perfect_rows.empty:
            perfect_rows["mean_cost"] = numeric(perfect_rows["mean_cost"])
            first = perfect_rows.iloc[0]
            if pd.notna(first.get("mean_cost")):
                perfect_cost = f"{float(first['mean_cost']):.2f}"
            completed = float(first.get("completed_count", 0) or 0)
            n_runs = float(first.get("n_runs", 0) or 0)
            perfect_vio = "0" if n_runs > 0 and completed == n_runs else "--"

    first_block = df[df["gamma_Q"].isin([0.0, 0.1, 0.3, 0.5])].copy()
    second_block = df[df["gamma_Q"].isin([0.7, 0.9, 1.0])].copy()
    if first_block.empty or second_block.empty:
        midpoint = max(1, int(math.ceil(len(df) / 2)))
        first_block = df.iloc[:midpoint].copy()
        second_block = df.iloc[midpoint:].copy()

    lines = [
        r"\begin{table}[!t]",
        r"\caption{Impact of the Scale Factor\label{tab:scale factor}}",
        r"\centering",
        r"\begin{tabular}{ccccc}",
        r"\toprule",
        r"Scale factor $\gamma_Q$ & " + " & ".join(gamma_label(v) for v in first_block["gamma_Q"]) + r" \\",
        r"\midrule",
        r"Ave. total cost (\$) & " + " & ".join(row_values(first_block, kind="cost")) + r" \\",
        r"Ave. per-step time (s) & " + " & ".join(row_values(first_block, kind="time")) + r" \\",
        r"Ave. cum. vio. (p.u.) & " + " & ".join(row_values(first_block, kind="vio")) + r" \\",
        r"\midrule",
        r"Scale factor $\gamma_Q$ & "
        + " & ".join([*(gamma_label(v) for v in second_block["gamma_Q"]), "Perfect"])
        + r" \\",
        r"\midrule",
        r"Ave. total cost (\$) & "
        + " & ".join([*row_values(second_block, kind="cost"), perfect_cost])
        + r" \\",
        r"Ave. per-step time (s) & "
        + " & ".join([*row_values(second_block, kind="time"), "--"])
        + r" \\",
        r"Ave. cum. vio. (p.u.) & "
        + " & ".join([*row_values(second_block, kind="vio"), perfect_vio])
        + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    for filename in ("table_33_gamma_validation.tex", "table_33_scale_factor_impact.tex"):
        path = table_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("Saved:", path)


def make_nn_table_tex(table_dir: Path) -> None:
    df = collect_33_nn_embedding_summary()

    def fmt_int(value: object) -> str:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(parsed):
            return "--"
        return f"{int(round(float(parsed)))}"

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Method": row["method"],
                "Cost ($)": fmt_mean_ci(row["mean_cost"], row["ci95_cost"], 2),
                "Total runtime (s)": fmt_mean_ci(
                    row["mean_solve_time_s"],
                    row["ci95_solve_time_s"],
                    2,
                ),
                "Vars.": fmt_int(row.get("mean_num_vars", np.nan)),
                "Bin. vars.": fmt_int(row.get("mean_num_bin_vars", np.nan)),
                "Constrs.": fmt_int(row.get("mean_num_constrs", np.nan)),
            }
        )
    tex_table(pd.DataFrame(rows), table_dir / "table_33_nn_embedding_compact.tex", "lccccc")


def make_model_complexity_table_tex(table_dir: Path) -> None:
    df = collect_33_model_complexity_summary()

    def fmt_int(value: object) -> str:
        parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(parsed):
            return "--"
        return f"{int(round(float(parsed)))}"

    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "Method": row["method"],
                "Cont. vars.": fmt_int(row.get("cont_vars", np.nan)),
                "Bin. vars.": fmt_int(row.get("bin_vars", np.nan)),
                "Constrs.": fmt_int(row.get("total_constrs", np.nan)),
            }
        )
    tex_table(pd.DataFrame(rows), table_dir / "table_33_model_complexity_compact.tex", "lccc")


def make_forecast_table_tex(table_dir: Path) -> None:
    df = read_csv("step5_33_pairwise_mpc3_vs_proposed_by_case.csv")
    assert df is not None
    df = df.copy()
    df["forecast_case"] = pd.Categorical(df["forecast_case"], FORECAST_CASE_ORDER, ordered=True)
    df = df.sort_values("forecast_case")

    rows = []
    for _, row in df.iterrows():
        case = str(row["forecast_case"])
        rows.append(
            {
                "Forecast case": FORECAST_CASE_LABELS.get(case, case),
                "Matched cases": int(row["n_matched"]),
                "Cost reduction (%)": fmt_mean_ci(
                    row["mean_cost_reduction_percent"],
                    row["ci95_cost_reduction_percent"],
                    2,
                ),
                "Time reduction (%)": fmt_mean_ci(
                    row["mean_time_reduction_percent"],
                    row["ci95_time_reduction_percent"],
                    2,
                ),
            }
        )
    tex_table(pd.DataFrame(rows), table_dir / "table_33_forecast_sensitivity_compact.tex", "lccc")


def write_manifest(figure_dir: Path, table_dir: Path, image_dir: Optional[Path]) -> None:
    figure_names = [
        "cost33.pdf",
        "time33.pdf",
        "gamma33.pdf",
        "forecast_sensitivity33.pdf",
        "nn_embedding33.pdf",
        "scheduling_results_P_pro.pdf",
        "scheduling_results_P_sac.pdf",
        "scheduling_results_P_mpc.pdf",
    ]
    table_names = [
        "table_33_main_compact.tex",
        "table_33_gamma_validation.tex",
        "table_33_scale_factor_impact.tex",
        "table_33_nn_embedding_compact.tex",
        "table_33_model_complexity_compact.tex",
        "table_33_forecast_sensitivity_compact.tex",
    ]
    extra_figure_paths = [
        RESULT_DIR / "paper_figures" / "119" / "scheduling_results_119_P_pro.pdf",
        RESULT_DIR / "paper_figures" / "119" / "scheduling_results_119_P_sac.pdf",
        RESULT_DIR / "paper_figures" / "119" / "scheduling_results_119_P_mpc.pdf",
    ]
    extra_table_paths = [
        RESULT_DIR / "paper_tables" / "33" / "table_33_icnn_value_error.md",
        RESULT_DIR / "paper_tables" / "33" / "table_33_icnn_decision_impact.md",
        RESULT_DIR / "paper_tables" / "33" / "mpc_diagnostic",
        RESULT_DIR / "paper_tables" / "33" / "mpc_diagnostic_perfect_forecast_H3",
        RESULT_DIR / "paper_tables" / "119" / "table_119_scalability.tex",
        RESULT_DIR / "paper_tables" / "119" / "table_119_model_complexity_compact.tex",
    ]

    def display_path(path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    lines = [
        "# Paper Output Manifest",
        "",
        "Generated from files under `results_v2/formal`.",
        "",
        "## Figures",
    ]
    for name in figure_names:
        path = figure_dir / name
        lines.append(f"- `{display_path(path)}`" + ("" if path.exists() else " (missing)"))
    for path in extra_figure_paths:
        lines.append(f"- `{display_path(path)}`" + ("" if path.exists() else " (missing)"))
    lines.extend(["", "## Tables"])
    for name in table_names:
        path = table_dir / name
        lines.append(f"- `{display_path(path)}`" + ("" if path.exists() else " (missing or skipped)"))
    for path in extra_table_paths:
        lines.append(f"- `{display_path(path)}`" + ("" if path.exists() else " (missing or skipped)"))
    if image_dir is not None:
        lines.extend(["", "## LaTeX Image Directory", f"- `{display_path(image_dir)}`"])

    manifest = RESULT_DIR / "paper_outputs_manifest.md"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved:", manifest)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-ready 33-bus figures and tables.")
    parser.add_argument("--figure-dir", default=str(DEFAULT_FIG_DIR))
    parser.add_argument("--table-dir", default=str(DEFAULT_TABLE_DIR))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument(
        "--no-image-copy",
        action="store_true",
        help="Do not copy generated PDFs/static images to the project-level images directory.",
    )
    parser.add_argument(
        "--skip-static-copy",
        action="store_true",
        help="Do not copy local static images from paper_assets/static_images.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    image_dir = None if args.no_image_copy else Path(args.image_dir)

    setup_matplotlib()
    ensure_dirs(figure_dir, table_dir)
    if image_dir is not None:
        ensure_dirs(image_dir)

    if not args.skip_static_copy:
        copy_static_images(image_dir)

    plot_cost33(figure_dir, image_dir)
    plot_time33(figure_dir, image_dir)
    plot_gamma_validation(figure_dir, image_dir)
    plot_forecast_sensitivity(figure_dir, image_dir)
    plot_nn_embedding(figure_dir, image_dir)
    plot_dispatch_schedules(figure_dir, image_dir)

    make_main_table_tex(table_dir)
    make_gamma_table_tex(table_dir)
    make_nn_table_tex(table_dir)
    make_model_complexity_table_tex(table_dir)
    make_forecast_table_tex(table_dir)
    write_manifest(figure_dir, table_dir, image_dir)

    print("")
    print("Done.")
    print("Figures:", figure_dir)
    print("Tables:", table_dir)
    if image_dir is not None:
        print("Images:", image_dir)


if __name__ == "__main__":
    main()
