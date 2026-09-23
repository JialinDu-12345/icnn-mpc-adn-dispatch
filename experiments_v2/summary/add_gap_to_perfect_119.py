from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


GROUP_KEYS = ["system", "actual_case", "forecast_case", "method", "horizon", "gamma_Q"]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return RESULT_DIR / path


def ci95(series: pd.Series) -> float:
    values = series.dropna()
    n = len(values)
    if n < 2:
        return math.nan
    return float(1.96 * values.std(ddof=1) / math.sqrt(n))


def parse_csv_list(raw: str) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        if item in seen:
            continue
        out.append(item)
        seen.add(item)
    return out


def parse_bool_series(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def normalize_key_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for key in GROUP_KEYS:
        if key not in out.columns:
            out[key] = ""
    out["system"] = out["system"].fillna("").astype(str)
    out["actual_case"] = out["actual_case"].fillna("").astype(str)
    out["forecast_case"] = out["forecast_case"].fillna("").astype(str)
    out["method"] = out["method"].fillna("").astype(str)
    out["horizon"] = out["horizon"].fillna("").astype(str)
    out["gamma_Q"] = out["gamma_Q"].fillna("").astype(str)
    return out


def read_result_csv(relative_path: str) -> pd.DataFrame:
    path = resolve_path(relative_path)
    if not path.exists():
        raise FileNotFoundError(path)
    print("[LOAD]", path)
    return pd.read_csv(path, dtype={"gamma_Q": str})


def completed_mask(df: pd.DataFrame) -> pd.Series:
    if "completed" in df.columns:
        return parse_bool_series(df["completed"])

    if "executed_steps" in df.columns:
        executed = pd.to_numeric(df["executed_steps"], errors="coerce").fillna(0)
    else:
        executed = pd.Series(0, index=df.index)

    if "solver_fail_count" in df.columns:
        failures = pd.to_numeric(df["solver_fail_count"], errors="coerce").fillna(0)
    else:
        failures = pd.Series(0, index=df.index)

    return (executed == 48) & (failures == 0)


def load_completed_raws(raw_files: Iterable[str]) -> pd.DataFrame:
    frames = []
    for relative_path in raw_files:
        df = read_result_csv(relative_path)
        df = normalize_key_columns(df)
        if "cost" not in df.columns:
            raise KeyError(f"Raw file is missing 'cost': {RESULT_DIR / relative_path}")
        if "scenario_id" not in df.columns:
            raise KeyError(f"Raw file is missing 'scenario_id': {resolve_path(relative_path)}")

        df = df[completed_mask(df)].copy()
        df["cost"] = pd.to_numeric(df["cost"], errors="coerce")
        df = df.dropna(subset=["cost"])
        df["scenario_id"] = df["scenario_id"].astype(str)
        df["source_raw"] = relative_path
        frames.append(df)

    if not frames:
        raise RuntimeError("No raw files were provided for 119 gap calculation.")

    raw = pd.concat(frames, ignore_index=True, sort=False)
    if raw.empty:
        raise RuntimeError("No completed 119 raw rows are available for gap calculation.")
    return raw


def extract_perfect(raw: pd.DataFrame) -> pd.DataFrame:
    perfect = raw[
        (raw["system"].astype(str) == "119")
        & (raw["method"].astype(str) == "Perfect-Global")
    ][["system", "actual_case", "scenario_id", "cost"]].copy()
    perfect = perfect.rename(columns={"cost": "perfect_cost"})
    perfect["perfect_cost"] = pd.to_numeric(perfect["perfect_cost"], errors="coerce")
    perfect = perfect.dropna(subset=["perfect_cost"])
    if perfect.empty:
        raise RuntimeError("No completed 119 Perfect-Global rows are available for gap calculation.")
    return perfect


def summarize_gaps(details: pd.DataFrame) -> pd.DataFrame:
    return (
        details.groupby(GROUP_KEYS, dropna=False)
        .agg(
            n_gap_matched=("scenario_id", "count"),
            mean_gap_to_perfect_percent=("gap_to_perfect_percent", "mean"),
            std_gap_to_perfect_percent=("gap_to_perfect_percent", "std"),
            ci95_gap_to_perfect_percent=("gap_to_perfect_percent", ci95),
        )
        .reset_index()
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Add scenario-matched 119-bus gap to Perfect-Global."
    )
    parser.add_argument(
        "--summary",
        default="formal/step10_119_scalability_inline20_summary.csv",
    )
    parser.add_argument(
        "--main-raw",
        default="formal/step10_119_main_inline20_raw.csv",
    )
    parser.add_argument(
        "--perfect-raw",
        default="formal/step10_119_main_inline20_raw.csv",
    )
    parser.add_argument(
        "--output",
        default="formal/step10_119_scalability_inline20_summary_with_gap.csv",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    summary = read_result_csv(args.summary)
    summary = normalize_key_columns(summary)
    raw = load_completed_raws(parse_csv_list(f"{args.main_raw},{args.perfect_raw}"))
    raw = raw.drop(columns=["perfect_cost"], errors="ignore")
    perfect = extract_perfect(raw)

    details = raw.merge(
        perfect,
        on=["system", "actual_case", "scenario_id"],
        how="inner",
    )
    if details.empty:
        raise RuntimeError(
            "No scenario-matched rows were found between 119 methods and Perfect-Global."
        )

    details["gap_to_perfect_percent"] = (
        (details["cost"] - details["perfect_cost"]) / details["perfect_cost"] * 100.0
    )

    gap_summary = summarize_gaps(details)
    out = summary.merge(gap_summary, on=GROUP_KEYS, how="left")

    output_path = resolve_path(args.output)
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
