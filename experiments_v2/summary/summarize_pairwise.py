from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


INDEX_COLS = ["system", "scenario_id", "actual_case", "forecast_case", "split"]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    if not clean:
        return math.nan
    return mean(clean)


def std_or_nan(values: Iterable[float]) -> float:
    clean = finite(values)
    if len(clean) < 2:
        return math.nan
    return stdev(clean)


def ci95(values: Iterable[float]) -> float:
    clean = finite(values)
    if len(clean) < 2:
        return math.nan
    return 1.96 * stdev(clean) / math.sqrt(len(clean))


def safe_percent_reduction(baseline: float, target: float) -> float:
    if not math.isfinite(baseline) or not math.isfinite(target) or baseline == 0:
        return math.nan
    return (baseline - target) / baseline * 100.0


def row_completed(row: Dict[str, str]) -> bool:
    if row.get("completed") not in (None, ""):
        return parse_bool(row.get("completed"))
    return (
        int(float(row.get("executed_steps") or 0)) == 48
        and int(float(row.get("solver_fail_count") or 0)) == 0
    )


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def scenario_key(row: Dict[str, str]) -> Tuple[str, ...]:
    return tuple(row.get(col, "") for col in INDEX_COLS)


def build_completed_lookup(
    rows: List[Dict[str, str]]
) -> Dict[Tuple[str, ...], Dict[str, Dict[str, str]]]:
    lookup: Dict[Tuple[str, ...], Dict[str, Dict[str, str]]] = {}
    for row in rows:
        if not row_completed(row):
            continue
        method = row.get("method", "")
        if not method:
            continue
        lookup.setdefault(scenario_key(row), {}).setdefault(method, row)
    return lookup


def build_pairwise_rows(
    rows: List[Dict[str, str]], baseline: str, target: str
) -> List[Dict[str, object]]:
    lookup = build_completed_lookup(rows)
    details: List[Dict[str, object]] = []

    for key in sorted(lookup):
        by_method = lookup[key]
        if baseline not in by_method or target not in by_method:
            continue

        baseline_row = by_method[baseline]
        target_row = by_method[target]
        baseline_cost = parse_float(baseline_row.get("cost"))
        target_cost = parse_float(target_row.get("cost"))
        baseline_time = parse_float(baseline_row.get("solve_time_s"))
        target_time = parse_float(target_row.get("solve_time_s"))

        cost_diff = target_cost - baseline_cost
        time_diff = target_time - baseline_time
        details.append(
            {
                **dict(zip(INDEX_COLS, key)),
                "baseline": baseline,
                "target": target,
                "baseline_horizon": baseline_row.get("horizon", ""),
                "target_horizon": target_row.get("horizon", ""),
                "baseline_gamma_Q": baseline_row.get("gamma_Q", ""),
                "target_gamma_Q": target_row.get("gamma_Q", ""),
                "baseline_cost": baseline_cost,
                "target_cost": target_cost,
                "cost_diff": cost_diff,
                "cost_reduction_percent": safe_percent_reduction(baseline_cost, target_cost),
                "baseline_solve_time_s": baseline_time,
                "target_solve_time_s": target_time,
                "time_diff_s": time_diff,
                "time_reduction_percent": safe_percent_reduction(baseline_time, target_time),
            }
        )
    return details


def summarize(details: List[Dict[str, object]], baseline: str, target: str) -> Dict[str, object]:
    cost_diffs = [parse_float(row.get("cost_diff")) for row in details]
    cost_reductions = [parse_float(row.get("cost_reduction_percent")) for row in details]
    time_diffs = [parse_float(row.get("time_diff_s")) for row in details]
    time_reductions = [parse_float(row.get("time_reduction_percent")) for row in details]

    return {
        "baseline": baseline,
        "target": target,
        "n_matched": len(details),
        "mean_cost_diff": mean_or_nan(cost_diffs),
        "std_cost_diff": std_or_nan(cost_diffs),
        "ci95_cost_diff": ci95(cost_diffs),
        "mean_cost_reduction_percent": mean_or_nan(cost_reductions),
        "std_cost_reduction_percent": std_or_nan(cost_reductions),
        "ci95_cost_reduction_percent": ci95(cost_reductions),
        "mean_time_diff_s": mean_or_nan(time_diffs),
        "std_time_diff_s": std_or_nan(time_diffs),
        "ci95_time_diff_s": ci95(time_diffs),
        "mean_time_reduction_percent": mean_or_nan(time_reductions),
        "std_time_reduction_percent": std_or_nan(time_reductions),
        "ci95_time_reduction_percent": ci95(time_reductions),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Matched-pair summary for 33-bus model-based experiment.")
    parser.add_argument("--input", default="formal/step4_33_main_model_based_raw.csv")
    parser.add_argument("--output", default="formal/step4_33_pairwise_mpc3_vs_proposed.csv")
    parser.add_argument("--baseline", default="MPC-3")
    parser.add_argument("--target", default="Proposed-H3")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    raw_path = RESULT_DIR / args.input
    rows = read_rows(raw_path)
    detail_rows = build_pairwise_rows(rows, args.baseline, args.target)
    summary_row = summarize(detail_rows, args.baseline, args.target)

    output_path = RESULT_DIR / args.output
    detail_path = output_path.with_name(f"{output_path.stem}_details{output_path.suffix or '.csv'}")

    summary_fields = list(summary_row.keys())
    detail_fields = [
        *INDEX_COLS,
        "baseline",
        "target",
        "baseline_horizon",
        "target_horizon",
        "baseline_gamma_Q",
        "target_gamma_Q",
        "baseline_cost",
        "target_cost",
        "cost_diff",
        "cost_reduction_percent",
        "baseline_solve_time_s",
        "target_solve_time_s",
        "time_diff_s",
        "time_reduction_percent",
    ]
    write_rows(output_path, [summary_row], summary_fields)
    write_rows(detail_path, detail_rows, detail_fields)

    print(summary_row)
    print("Saved:", output_path)
    print("Saved details:", detail_path)


if __name__ == "__main__":
    main()
