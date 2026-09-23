from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


RAW_CANDIDATES = [
    "formal/step7b_33_perfect_global_raw.csv",
    "formal/step7_33_oracle_mpc_raw.csv",
    "formal/step4_33_main_model_based_raw.csv",
    "formal/step6_33_stdnn_only_raw.csv",
    "formal/step6_33_nn_embedding_raw.csv",
    "formal/step7_33_policy_baselines_raw.csv",
    "formal/step7_33_saclag_stochastic_seed0_test_raw.csv",
]

GROUP_KEYS = ["system", "actual_case", "forecast_case", "method", "horizon", "gamma_Q"]


def ci95(series: pd.Series) -> float:
    values = series.dropna()
    n = len(values)
    if n < 2:
        return math.nan
    return float(1.96 * values.std(ddof=1) / math.sqrt(n))


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for key in GROUP_KEYS:
        if key not in out.columns:
            out[key] = ""
    out["system"] = out["system"].astype(str)
    out["actual_case"] = out["actual_case"].fillna("").astype(str)
    out["forecast_case"] = out["forecast_case"].fillna("").astype(str)
    out["method"] = out["method"].fillna("").astype(str)
    out["horizon"] = out["horizon"].fillna("").astype(str)
    out["gamma_Q"] = out["gamma_Q"].fillna("").astype(str)
    return out


def read_csv_if_exists(relative_path: str) -> pd.DataFrame | None:
    path = RESULT_DIR / relative_path
    if not path.exists():
        print("[MISS]", path)
        return None
    print("[LOAD]", path)
    return pd.read_csv(path, dtype={"gamma_Q": str})


def load_completed_perfect(perfect_raw: str, require_certified: bool) -> pd.DataFrame:
    perfect = read_csv_if_exists(perfect_raw)
    if perfect is None:
        raise FileNotFoundError(RESULT_DIR / perfect_raw)

    if "completed" not in perfect.columns:
        raise KeyError(f"Perfect raw is missing 'completed': {RESULT_DIR / perfect_raw}")
    perfect["completed_bool"] = parse_bool_series(perfect["completed"])
    mask = (perfect["method"] == "Perfect-Global") & perfect["completed_bool"]

    if require_certified:
        if "certified_global" not in perfect.columns:
            raise KeyError(
                "Perfect raw does not contain certified_global. "
                "Regenerate it with the current run_33_perfect_global.py."
            )
        mask = mask & parse_bool_series(perfect["certified_global"])

    perfect = perfect.loc[mask, ["system", "actual_case", "scenario_id", "cost"]].copy()
    perfect = perfect.rename(columns={"cost": "perfect_cost"})
    perfect["system"] = perfect["system"].fillna("").astype(str)
    perfect["actual_case"] = perfect["actual_case"].fillna("").astype(str)
    perfect["scenario_id"] = perfect["scenario_id"].astype(str)
    perfect["perfect_cost"] = pd.to_numeric(perfect["perfect_cost"], errors="coerce")
    perfect = perfect.dropna(subset=["perfect_cost"])
    if perfect.empty:
        raise RuntimeError("No completed Perfect-Global rows are available for gap calculation.")
    return perfect


def load_completed_raws(raw_files: List[str]) -> pd.DataFrame:
    frames = []
    for relative_path in raw_files:
        df = read_csv_if_exists(relative_path)
        if df is None:
            continue
        if "completed" not in df.columns:
            print("[SKIP] completed column missing:", relative_path)
            continue
        df = normalize_key_columns(df)
        df["completed_bool"] = parse_bool_series(df["completed"])
        df = df[df["completed_bool"]].copy()
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
        df = df.dropna(subset=["cost"])
        df["scenario_id"] = df["scenario_id"].astype(str)
        df["source_raw"] = relative_path
        if "saclag_stochastic_seed0_test" in relative_path:
            df = df[df["method"].eq("SACLag")].copy()
            df["source_rank"] = -10
        elif "step6_33_stdnn_only_raw" in relative_path:
            df = df[df["method"].eq("StdNN-H3-MIP")].copy()
            df["source_rank"] = -5
        elif "step6_33_nn_embedding_raw" in relative_path:
            df = df[df["method"].eq("StdNN-H3-MIP")].copy()
            df["source_rank"] = -4
        else:
            df["source_rank"] = 0
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No completed raw files found for matched gap calculation.")
    combined = pd.concat(frames, ignore_index=True, sort=False)
    duplicate_columns = [*GROUP_KEYS, "scenario_id"]
    combined = combined.sort_values([*duplicate_columns, "source_rank"])
    return combined.drop_duplicates(subset=duplicate_columns, keep="first")


def summarize_gaps(details: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        details.groupby(GROUP_KEYS, dropna=False)
        .agg(
            n_gap_matched=("scenario_id", "count"),
            mean_gap_to_perfect_percent=("gap_to_perfect_percent", "mean"),
            std_gap_to_perfect_percent=("gap_to_perfect_percent", "std"),
            ci95_gap_to_perfect_percent=("gap_to_perfect_percent", ci95),
        )
        .reset_index()
    )
    return grouped


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add scenario-matched gap to Perfect-Global.")
    parser.add_argument("--input", default="formal/step8_33_main_table_with_perfect.csv")
    parser.add_argument("--perfect-raw", default="formal/step7b_33_perfect_global_raw.csv")
    parser.add_argument("--raw-files", default=",".join(RAW_CANDIDATES))
    parser.add_argument("--output", default="formal/step8_33_main_table_with_perfect_gap.csv")
    parser.add_argument(
        "--require-certified",
        action="store_true",
        help="Use only Perfect-Global rows with certified_global=True.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    summary_path = RESULT_DIR / args.input
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)

    summary = pd.read_csv(summary_path, dtype={"gamma_Q": str})
    summary = normalize_key_columns(summary)
    perfect = load_completed_perfect(args.perfect_raw, require_certified=args.require_certified)
    raw_files = [item.strip() for item in args.raw_files.split(",") if item.strip()]
    raw = load_completed_raws(raw_files)
    raw = raw.drop(columns=["perfect_cost"], errors="ignore")

    details = raw.merge(
        perfect,
        on=["system", "actual_case", "scenario_id"],
        how="inner",
    )
    details["gap_to_perfect_percent"] = (
        (details["cost"] - details["perfect_cost"]) / details["perfect_cost"] * 100.0
    )

    gap_summary = summarize_gaps(details)
    out = summary.merge(gap_summary, on=GROUP_KEYS, how="left")

    output_path = RESULT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False, encoding="utf-8-sig")

    detail_path = output_path.with_name(f"{output_path.stem}_details.csv")
    details.to_csv(detail_path, index=False, encoding="utf-8-sig")

    display_cols = [
        "method",
        "forecast_case",
        "mean_cost",
        "n_gap_matched",
        "mean_gap_to_perfect_percent",
    ]
    existing = [col for col in display_cols if col in out.columns]
    print(out[existing])
    print("Saved:", output_path)
    print("Saved details:", detail_path)


if __name__ == "__main__":
    main()
