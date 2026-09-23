from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


METHOD_ORDER = [
    "Perfect-Global",
    "Oracle-H3",
    "Oracle-H7",
    "Oracle-H12",
    "MPC-3",
    "MPC-5",
    "MPC-7",
    "StdNN-H3-MIP",
    "SAC",
    "SAC-ICNN",
    "SACLag",
    "Proposed-H3",
]


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    out = df[df["method"].isin(METHOD_ORDER)].copy()
    out["method_order"] = out["method"].map({name: idx for idx, name in enumerate(METHOD_ORDER)})
    return out.sort_values(["method_order", "method"])


def numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([float("nan")] * len(df), index=df.index)
    return pd.to_numeric(df[column], errors="coerce")


def save_bar(
    df: pd.DataFrame,
    *,
    value_col: str,
    err_col: str | None,
    ylabel: str,
    output_path: Path,
    log_scale: bool = False,
) -> None:
    plot_df = ordered(df)
    values = numeric(plot_df, value_col)
    yerr = numeric(plot_df, err_col) if err_col else None

    fig, ax = plt.subplots(figsize=(8.8, 3.8))
    ax.bar(plot_df["method"], values, yerr=yerr, capsize=3 if err_col else 0)
    if log_scale:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment("right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Saved:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 33-bus main-result figures.")
    parser.add_argument("--input", default="formal/step8_33_main_table_with_perfect_gap.csv")
    parser.add_argument("--output-dir", default="paper_figures/33")
    args = parser.parse_args()

    df = pd.read_csv(RESULT_DIR / args.input, dtype={"gamma_Q": str})
    out_dir = RESULT_DIR / args.output_dir

    save_bar(
        df,
        value_col="mean_cost",
        err_col="ci95_cost",
        ylabel="Mean daily cost",
        output_path=out_dir / "fig_33_main_cost.png",
    )
    save_bar(
        df,
        value_col="mean_solve_time_s",
        err_col="ci95_solve_time_s",
        ylabel="Solve time (s, log scale)",
        output_path=out_dir / "fig_33_main_runtime.png",
        log_scale=True,
    )
    if "mean_gap_to_perfect_percent" in df.columns:
        save_bar(
            df,
            value_col="mean_gap_to_perfect_percent",
            err_col="ci95_gap_to_perfect_percent",
            ylabel="Gap to Perfect-Global (%)",
            output_path=out_dir / "fig_33_main_gap_to_perfect.png",
        )


if __name__ == "__main__":
    main()
