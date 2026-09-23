from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


FORECAST_ORDER = ["id_reliable", "high_error", "biased_ood"]
METHOD_ORDER = ["MPC-3", "MPC-7", "Proposed-H3"]


def plot_grouped_cost(df: pd.DataFrame, output_path: Path) -> None:
    df = df[df["method"].isin(METHOD_ORDER)].copy()
    pivot = df.pivot_table(index="forecast_case", columns="method", values="mean_cost", aggfunc="mean")
    pivot = pivot.reindex(FORECAST_ORDER)
    pivot = pivot[[method for method in METHOD_ORDER if method in pivot.columns]]

    ax = pivot.plot(kind="bar", figsize=(7.2, 3.6), rot=0)
    ax.set_ylabel("Mean daily cost")
    ax.set_xlabel("Forecast case")
    ax.legend(ncol=3, frameon=False)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print("Saved:", output_path)


def plot_pairwise_reduction(pairwise: pd.DataFrame, output_path: Path) -> None:
    pairwise = pairwise.copy()
    pairwise["forecast_order"] = pairwise["forecast_case"].map(
        {name: idx for idx, name in enumerate(FORECAST_ORDER)}
    )
    pairwise = pairwise.sort_values("forecast_order")
    yerr = (
        pd.to_numeric(pairwise["ci95_cost_reduction_percent"], errors="coerce")
        if "ci95_cost_reduction_percent" in pairwise.columns
        else None
    )

    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.bar(
        pairwise["forecast_case"],
        pd.to_numeric(pairwise["mean_cost_reduction_percent"], errors="coerce"),
        yerr=yerr,
        capsize=3 if yerr is not None else 0,
    )
    ax.axhline(0.0, linewidth=0.8, color="black")
    ax.set_ylabel("Cost reduction vs baseline (%)")
    ax.set_xlabel("Forecast case")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print("Saved:", output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 33-bus forecast-sensitivity figures.")
    parser.add_argument("--summary", default="formal/step5_33_forecast_sensitivity_summary.csv")
    parser.add_argument("--pairwise", default="formal/step5_33_pairwise_mpc3_vs_proposed_by_case.csv")
    parser.add_argument("--output-dir", default="paper_figures/33")
    args = parser.parse_args()

    out_dir = RESULT_DIR / args.output_dir
    summary_path = RESULT_DIR / args.summary
    if summary_path.exists():
        plot_grouped_cost(pd.read_csv(summary_path), out_dir / "fig_33_forecast_sensitivity_cost.png")
    else:
        print("[MISS]", summary_path)

    pairwise_path = RESULT_DIR / args.pairwise
    if pairwise_path.exists():
        plot_pairwise_reduction(
            pd.read_csv(pairwise_path),
            out_dir / "fig_33_forecast_sensitivity_reduction.png",
        )
    else:
        print("[MISS]", pairwise_path)


if __name__ == "__main__":
    main()
