from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import (              
    GAMMA_Q_CANDIDATES_33,
    SPLITS,
    validate_actual_case,
    validate_forecast_case,
)
from experiments_v2.result_logger import ResultLogger              
from experiments_v2.run_33_mpc import run_one_scenario_33              


def parse_float_list(raw: Optional[str], default: List[float]) -> List[float]:
    if not raw:
        return list(default)
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def parse_scenario_ids(raw: Optional[str], scenario_count: Optional[int]) -> List[int]:
    if raw:
        normalized = raw.strip().lower()
        if normalized in {"val", "validation"}:
            scenario_ids = list(SPLITS["33"]["val"])
        else:
            scenario_ids = [int(item.strip()) for item in raw.split(",") if item.strip()]
    else:
        scenario_ids = list(SPLITS["33"]["val"])
    if scenario_count is not None:
        return scenario_ids[:scenario_count]
    return scenario_ids


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run 33-bus Proposed-H3 gamma_Q validation on validation scenarios."
    )
    parser.add_argument("--scenarios", default=None, help="Comma-separated validation IDs.")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument(
        "--gamma-candidates",
        default="0,0.1,0.3,0.5,0.7,0.9,1.0",
        help="Comma-separated gamma_Q candidates. If omitted, use GAMMA_Q_CANDIDATES_33.",
    )
    parser.add_argument(
        "--gamma-values",
        default=None,
        help="Backward-compatible alias for --gamma-candidates.",
    )
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-tail", action="store_true")
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step3_33_gamma_validation_raw.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)

    scenario_ids = parse_scenario_ids(args.scenarios, args.scenario_count)
    gamma_raw = args.gamma_candidates if args.gamma_candidates is not None else args.gamma_values
    gamma_values = parse_float_list(gamma_raw, GAMMA_Q_CANDIDATES_33)

    if args.dry_run:
        print("Dry-run gamma validation")
        print("scenarios:", scenario_ids)
        print("gamma_values:", gamma_values)
        print("forecast_case:", args.forecast_case)
        print("actual_case:", args.actual_case)
        print("max_steps:", args.max_steps)
        print("include_tail:", not args.no_tail)
        return

    logger = ResultLogger(args.output)
    for gamma_q in gamma_values:
        for scenario_id in scenario_ids:
            if abs(float(gamma_q)) <= 1e-12:
                method_name = "MPC-3"
                solver_kind = "pure_mpc_clean"
                gamma_for_solver = None
            else:
                method_name = "Proposed-H3-gamma"
                solver_kind = "proposed"
                gamma_for_solver = float(gamma_q)

            row = run_one_scenario_33(
                method=method_name,
                solver_kind=solver_kind,
                scenario_id=scenario_id,
                forecast_case=args.forecast_case,
                actual_case=args.actual_case,
                horizon=3,
                gamma_Q=gamma_for_solver,
                split="val",
                max_steps=args.max_steps,
                include_tail=not args.no_tail,
                quiet_solver=not args.verbose_solver,
            )
            row["gamma_Q"] = float(gamma_q)
            row["gamma_validation_model"] = method_name
            logger.add_row(row)
            print(
                f"gamma={gamma_q}, scenario={scenario_id}, "
                f"completed={row.get('completed')}, "
                f"reason={row.get('incomplete_reason')}, "
                f"steps={row.get('executed_steps')}, "
                f"cost={float(row.get('cost') or 0.0):.4f}, "
                f"time={float(row.get('solve_time_s') or 0.0):.2f}, "
                f"fail={row.get('solver_fail_count')}"
            )

    path = logger.save()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
