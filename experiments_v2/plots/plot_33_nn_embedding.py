from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


METHOD_ORDER = ["MPC-3", "StdNN-H3-MIP", "Proposed-H3"]
MODEL_SIZE_FIELDS = ["mean_num_vars", "mean_num_bin_vars", "mean_num_constrs"]


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["method"].isin(METHOD_ORDER)].copy()
    out["method_order"] = out["method"].map({name: idx for idx, name in enumerate(METHOD_ORDER)})
    return out.sort_values("method_order")


def missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def read_optional(relative_path: str | None) -> pd.DataFrame | None:
    if not relative_path:
        return None
    path = RESULT_DIR / relative_path
    if not path.exists():
        print("[MISS]", path)
        return None
    return pd.read_csv(path, dtype={"gamma_Q": str})


def first_method_row(df: pd.DataFrame | None, method: str) -> pd.Series | None:
    if df is None or "method" not in df.columns:
        return None
    rows = df[df["method"].astype(str).eq(method)].copy()
    if rows.empty:
        return None
    if "n_runs" in rows.columns:
        rows["_n_runs_sort"] = pd.to_numeric(rows["n_runs"], errors="coerce").fillna(-1)
        rows = rows.sort_values("_n_runs_sort", ascending=False)
    return rows.iloc[0]


def combine_summaries(
    nn_summary: pd.DataFrame | None,
    main_summary: pd.DataFrame | None,
    stdnn_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    for method in METHOD_ORDER:
        if method == "StdNN-H3-MIP":
            perf_sources = [stdnn_summary, nn_summary, main_summary]
            model_sources = [stdnn_summary, nn_summary, main_summary]
        else:
            perf_sources = [main_summary, nn_summary]
            model_sources = [nn_summary, main_summary]

        perf_row = next(
            (row for row in (first_method_row(df, method) for df in perf_sources) if row is not None),
            None,
        )
        if perf_row is None:
            continue
        merged = perf_row.to_dict()
        model_row = next(
            (row for row in (first_method_row(df, method) for df in model_sources) if row is not None),
            None,
        )
        if model_row is not None:
            for field in MODEL_SIZE_FIELDS:
                if field in model_row and missing(merged.get(field)):
                    merged[field] = model_row[field]
        rows.append(merged)
    return ordered(pd.DataFrame(rows))


def bar_plot(
    df: pd.DataFrame,
    *,
    value_col: str,
    err_col: str | None,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    plot_df = ordered(df)
    yerr = (
        pd.to_numeric(plot_df[err_col], errors="coerce")
        if err_col and err_col in plot_df.columns
        else None
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(
        plot_df["method"],
        pd.to_numeric(plot_df[value_col], errors="coerce"),
        yerr=yerr,
        capsize=3 if yerr is not None else 0,
    )
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Saved:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 33-bus NN-embedding figures.")
    parser.add_argument("--input", default="formal/step6_33_nn_embedding_summary.csv")
    parser.add_argument("--main-summary", default="formal/step4_33_main_model_based_summary.csv")
    parser.add_argument("--stdnn-summary", default="formal/step6_33_stdnn_only_summary.csv")
    parser.add_argument("--output-dir", default="paper_figures/33")
    args = parser.parse_args()

    df = combine_summaries(
        read_optional(args.input),
        read_optional(args.main_summary),
        read_optional(args.stdnn_summary),
    )
    out_dir = RESULT_DIR / args.output_dir
    bar_plot(
        df,
        value_col="mean_solve_time_s",
        err_col="ci95_solve_time_s",
        ylabel="Runtime (s, log scale)",
        output_path=out_dir / "fig_33_nn_embedding_runtime.png",
        log_scale=True,
    )
    bar_plot(
        df,
        value_col="mean_num_bin_vars",
        err_col=None,
        ylabel="Number of binary variables",
        output_path=out_dir / "fig_33_nn_embedding_binary_vars.png",
    )


if __name__ == "__main__":
    main()
