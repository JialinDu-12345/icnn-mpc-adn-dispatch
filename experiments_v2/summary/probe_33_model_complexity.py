from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR, validate_actual_case, validate_forecast_case              
from experiments_v2.run_33_mpc import load_selected_gamma, run_one_scenario_33              


TASK_SPECS: Dict[str, Dict[str, Any]] = {
    "MPC-3": {
        "method": "MPC-3",
        "solver_kind": "pure_mpc_clean",
        "horizon": 3,
        "gamma_Q": None,
    },
    "MPC-5": {
        "method": "MPC-5",
        "solver_kind": "pure_mpc_clean",
        "horizon": 5,
        "gamma_Q": None,
    },
    "MPC-7": {
        "method": "MPC-7",
        "solver_kind": "pure_mpc_clean",
        "horizon": 7,
        "gamma_Q": None,
    },
    "StdNN-H3-MIP": {
        "method": "StdNN-H3-MIP",
        "solver_kind": "std_nn_mpc",
        "horizon": 3,
        "gamma_Q": "auto",
    },
    "Proposed-H3": {
        "method": "Proposed-H3",
        "solver_kind": "proposed",
        "horizon": 3,
        "gamma_Q": "auto",
    },
}

TASK_ORDER = ["MPC-3", "MPC-5", "MPC-7", "StdNN-H3-MIP", "Proposed-H3"]
TASK_ALIASES = {
    "mpc3": "MPC-3",
    "mpc-3": "MPC-3",
    "mpc5": "MPC-5",
    "mpc-5": "MPC-5",
    "mpc7": "MPC-7",
    "mpc-7": "MPC-7",
    "stdnn": "StdNN-H3-MIP",
    "std-nn": "StdNN-H3-MIP",
    "stdnn-h3": "StdNN-H3-MIP",
    "stdnn-h3-mip": "StdNN-H3-MIP",
    "proposed": "Proposed-H3",
    "proposed3": "Proposed-H3",
    "proposed-h3": "Proposed-H3",
}

OUTPUT_COLUMNS = [
    "system",
    "method",
    "scenario_id",
    "forecast_case",
    "actual_case",
    "horizon",
    "solver_window",
    "normal_step_index",
    "gamma_Q",
    "solver_kind",
    "cont_vars",
    "bin_vars",
    "total_vars",
    "linear_constrs",
    "quadratic_constrs",
    "general_constrs",
    "total_constrs",
    "gurobi_status",
    "sol_count",
    "mip_gap",
    "solve_time_s",
    "note",
]


def parse_tasks(raw: str) -> List[str]:
    if raw.strip().lower() == "all":
        return list(TASK_ORDER)

    names: List[str] = []
    seen = set()
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        name = TASK_ALIASES.get(item.lower(), item)
        if name not in TASK_SPECS:
            raise ValueError(f"Unknown task '{item}'. Available tasks: {TASK_ORDER}")
        if name in seen:
            continue
        names.append(name)
        seen.add(name)
    return names


def as_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return default
    return float(text)


def as_int(value: object, *, default: Optional[int] = None, field: str = "") -> int:
    numeric = as_float(value)
    if not math.isfinite(numeric):
        if default is not None:
            return int(default)
        raise RuntimeError(f"Missing finite model-complexity value for {field or 'field'}.")
    return int(round(numeric))


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No model-complexity rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def probe_one(
    *,
    task_name: str,
    scenario_id: int,
    forecast_case: str,
    actual_case: str,
    gamma_q: Optional[float],
    time_limit_s: Optional[float],
    mip_gap: Optional[float],
    verbose_solver: bool,
) -> Dict[str, object]:
    spec = TASK_SPECS[task_name]
    gamma_value = gamma_q if spec["gamma_Q"] == "auto" else spec["gamma_Q"]
    horizon = int(spec["horizon"])
    row = run_one_scenario_33(
        method=str(spec["method"]),
        solver_kind=str(spec["solver_kind"]),
        scenario_id=int(scenario_id),
        forecast_case=forecast_case,
        actual_case=actual_case,
        horizon=horizon,
        gamma_Q=gamma_value,
        split="test_id",
        max_steps=1,
        include_tail=False,
        quiet_solver=not verbose_solver,
        time_limit_s=time_limit_s,
        mip_gap=mip_gap,
    )
    if int(row.get("solver_fail_count") or 0) != 0:
        raise RuntimeError(f"Model-complexity probe failed for {task_name}: {row.get('failure_message')}")

    total_vars = as_int(row.get("mean_num_vars"), field="mean_num_vars")
    bin_vars = as_int(row.get("mean_num_bin_vars"), field="mean_num_bin_vars")
    linear_constrs = as_int(row.get("mean_num_constrs"), field="mean_num_constrs")
    quadratic_constrs = as_int(row.get("mean_num_qconstrs"), default=0)
    general_constrs = as_int(row.get("mean_num_genconstrs"), default=0)
    total_constrs = linear_constrs + quadratic_constrs + general_constrs

    return {
        "system": "33",
        "method": str(spec["method"]),
        "scenario_id": int(scenario_id),
        "forecast_case": forecast_case,
        "actual_case": actual_case,
        "horizon": horizon,
        "solver_window": int(row.get("solver_window") or horizon + 1),
        "normal_step_index": 0,
        "gamma_Q": "" if gamma_value is None else gamma_value,
        "solver_kind": str(spec["solver_kind"]),
        "cont_vars": total_vars - bin_vars,
        "bin_vars": bin_vars,
        "total_vars": total_vars,
        "linear_constrs": linear_constrs,
        "quadratic_constrs": quadratic_constrs,
        "general_constrs": general_constrs,
        "total_constrs": total_constrs,
        "gurobi_status": as_int(row.get("mean_model_status"), default=-1),
        "sol_count": as_int(row.get("min_model_sol_count"), default=0),
        "mip_gap": row.get("mean_mip_gap", ""),
        "solve_time_s": row.get("solve_time_s", ""),
        "note": (
            "One normal full-window MPC optimization model; tail/remainder "
            "windows are excluded. total_constrs = linear + quadratic + general."
        ),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe integer optimization-model sizes for 33-bus MPC-based methods."
    )
    parser.add_argument("--tasks", default="all", help="Comma-separated methods or 'all'.")
    parser.add_argument("--scenario", type=int, default=80)
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--gamma-Q", default="auto", help="Float value or 'auto'.")
    parser.add_argument("--time-limit-s", type=float, default=300)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step6_33_model_complexity_summary.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)

    task_names = parse_tasks(args.tasks)
    requires_gamma = any(TASK_SPECS[name]["gamma_Q"] == "auto" for name in task_names)
    if requires_gamma:
        gamma_q = load_selected_gamma("33") if str(args.gamma_Q).strip().lower() == "auto" else float(args.gamma_Q)
    else:
        gamma_q = None

    print("Model-complexity probe")
    print("Scenario:", args.scenario)
    print("Tasks:", task_names)
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("gamma_Q:", gamma_q if requires_gamma else "not required")
    print("Counts are taken from one normal full-window optimization model per method.")

    if args.dry_run:
        return

    rows = []
    for task_name in task_names:
        row = probe_one(
            task_name=task_name,
            scenario_id=args.scenario,
            forecast_case=args.forecast_case,
            actual_case=args.actual_case,
            gamma_q=gamma_q,
            time_limit_s=args.time_limit_s,
            mip_gap=args.mip_gap,
            verbose_solver=args.verbose_solver,
        )
        rows.append(row)
        print(
            f"{row['method']}: cont={row['cont_vars']}, bin={row['bin_vars']}, "
            f"total_vars={row['total_vars']}, total_constrs={row['total_constrs']} "
            f"(lin={row['linear_constrs']}, q={row['quadratic_constrs']}, "
            f"gen={row['general_constrs']})"
        )

    output_path = RESULT_DIR / args.output
    write_rows(output_path, rows)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
