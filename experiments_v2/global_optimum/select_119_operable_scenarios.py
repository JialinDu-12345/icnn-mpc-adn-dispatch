from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.global_optimum.run_119_perfect_global import (              
    run_one_perfect_global_119,
)
from experiments_v2.global_optimum.select_119_perfect_feasible import (              
    fieldnames_for_rows,
    is_selected_perfect_row,
    load_existing_rows,
    parse_bool,
    parse_float,
    parse_int,
    scenario_id_from_row,
    sort_rows_by_scenario,
    write_rows,
)
from experiments_v2.run_119_mpc import run_one_scenario_119              
from experiments_v2.utils.scenario_selection import (              
    parse_csv_scenarios,
    resolve_result_path,
)
from experiments_v2.config import (              
    DEFAULT_119,
    RAW_RESULT_COLUMNS,
    validate_actual_case,
    validate_forecast_case,
)


SELECTION_RULE = "operable_perfect_global_and_rolling_mpc3_subset"
SCREENING_RULE = "perfect_global_certified_and_rolling_mpc3_completed"


def method_from_row(row: Dict[str, Any]) -> str:
    return str(row.get("method", "")).strip()


def key_from_row(row: Dict[str, Any]) -> Tuple[int, str]:
    return scenario_id_from_row(row), method_from_row(row)


def rows_by_scenario_method(rows: List[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    out: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in rows:
        out[key_from_row(row)] = row
    return out


def is_selected_mpc3_row(row: Dict[str, Any]) -> bool:
    if method_from_row(row) != "MPC-3":
        return False
    completed = parse_bool(row.get("completed"))
    fail_count = parse_int(row.get("solver_fail_count"), default=0)
    timeout_count = parse_int(row.get("timeout_count"), default=0)
    executed_steps = parse_int(row.get("executed_steps"), default=0)
    incomplete_reason = str(row.get("incomplete_reason", "")).strip()
    return (
        completed
        and fail_count == 0
        and timeout_count == 0
        and executed_steps == int(DEFAULT_119["n_steps"])
        and incomplete_reason == ""
    )


def selected_scenario_ids(
    rows: List[Dict[str, Any]],
    *,
    mip_gap_target: float,
    target_count: int,
) -> List[int]:
    row_map = rows_by_scenario_method(rows)
    candidate_ids = sorted({scenario_id_from_row(row) for row in rows})
    selected: List[int] = []
    for scenario_id in candidate_ids:
        perfect = row_map.get((scenario_id, "Perfect-Global"))
        mpc3 = row_map.get((scenario_id, "MPC-3"))
        if perfect is None or mpc3 is None:
            continue
        if not is_selected_perfect_row(perfect, mip_gap_target=mip_gap_target):
            continue
        if not is_selected_mpc3_row(mpc3):
            continue
        selected.append(scenario_id)
        if len(selected) >= target_count:
            break
    return selected


def mark_screening_row(
    row: Dict[str, Any],
    *,
    candidate_order: int,
    role: str,
    selected_json: str,
    selected: bool,
) -> Dict[str, Any]:
    out = dict(row)
    out["operable_screening_candidate_order"] = int(candidate_order)
    out["operable_screening_rule"] = SCREENING_RULE
    out["operable_screening_role"] = role
    out["scenario_selection_file"] = selected_json if selected else ""
    out["scenario_selection_rule"] = SELECTION_RULE if selected else "operable_screening_candidate"
    return out


def split_selected_rows(
    rows: List[Dict[str, Any]],
    selected_ids: List[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    selected_set = set(selected_ids)
    perfect_rows: List[Dict[str, Any]] = []
    mpc_rows: List[Dict[str, Any]] = []
    for row in sort_rows_by_scenario(rows):
        sid = scenario_id_from_row(row)
        if sid not in selected_set:
            continue
        if method_from_row(row) == "Perfect-Global":
            perfect_rows.append(row)
        elif method_from_row(row) == "MPC-3":
            mpc_rows.append(row)
    return perfect_rows, mpc_rows


def save_selection_json(
    *,
    path: Path,
    selected_scenarios: List[int],
    candidate_scenarios: List[int],
    scanned_scenarios: List[int],
    target_count: int,
    actual_case: str,
    forecast_case: str,
    mip_gap: float,
    scan_output: str,
    selected_perfect_output: str,
    selected_mpc_output: str,
) -> None:
    payload = {
        "system": "119",
        "selection_rule": SELECTION_RULE,
        "screening_rule": SCREENING_RULE,
        "actual_case": actual_case,
        "forecast_case": forecast_case,
        "target_count": int(target_count),
        "selected_count": len(selected_scenarios),
        "selected_scenarios": selected_scenarios,
        "candidate_scenarios": candidate_scenarios,
        "scanned_scenarios": scanned_scenarios,
        "n_candidate_scenarios": len(candidate_scenarios),
        "n_candidates_scanned": len(scanned_scenarios),
        "mip_gap_target": float(mip_gap),
        "scan_output": scan_output,
        "selected_perfect_output": selected_perfect_output,
        "selected_mpc_output": selected_mpc_output,
        "note": (
            "Scenarios are selected before Proposed-H3 is evaluated. A scenario "
            "must be feasible/certified under full-day Perfect-Global and must "
            "also complete a 48-step rolling MPC-3 run under the formal forecast "
            "protocol."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved selected scenario JSON:", path)


def write_screening_outputs(
    *,
    scan_path: Path,
    selected_perfect_path: Path,
    selected_mpc_path: Path,
    json_path: Path,
    scan_rows: List[Dict[str, Any]],
    selected_ids: List[int],
    candidate_scenarios: List[int],
    target_count: int,
    actual_case: str,
    forecast_case: str,
    mip_gap: float,
    scan_output: str,
    selected_perfect_output: str,
    selected_mpc_output: str,
) -> None:
    sorted_rows = sort_rows_by_scenario(scan_rows)
    perfect_rows, mpc_rows = split_selected_rows(sorted_rows, selected_ids)
    write_rows(scan_path, sorted_rows)
    write_rows(selected_perfect_path, perfect_rows)
    write_rows(selected_mpc_path, mpc_rows)
    scanned_ids = sorted({scenario_id_from_row(row) for row in scan_rows})
    save_selection_json(
        path=json_path,
        selected_scenarios=selected_ids,
        candidate_scenarios=candidate_scenarios,
        scanned_scenarios=scanned_ids,
        target_count=target_count,
        actual_case=actual_case,
        forecast_case=forecast_case,
        mip_gap=mip_gap,
        scan_output=scan_output,
        selected_perfect_output=selected_perfect_output,
        selected_mpc_output=selected_mpc_output,
    )


def seed_perfect_rows(
    scan_rows: List[Dict[str, Any]],
    *,
    seed_path: Path,
    candidate_scenarios: List[int],
    selected_json: str,
) -> List[Dict[str, Any]]:
    if not seed_path.exists():
        print(f"[SKIP] Perfect seed scan not found: {seed_path}")
        return scan_rows

    order_map = {int(scenario_id): idx for idx, scenario_id in enumerate(candidate_scenarios)}
    existing_keys = set(rows_by_scenario_method(scan_rows))
    seeded_count = 0
    for row in load_existing_rows(seed_path):
        if method_from_row(row) != "Perfect-Global":
            continue
        sid = scenario_id_from_row(row)
        if sid not in order_map:
            continue
        key = (sid, "Perfect-Global")
        if key in existing_keys:
            continue
        scan_rows.append(
            mark_screening_row(
                row,
                candidate_order=order_map[sid],
                role="perfect_global",
                selected_json=selected_json,
                selected=False,
            )
        )
        existing_keys.add(key)
        seeded_count += 1
    print(f"Seeded {seeded_count} Perfect-Global rows from: {seed_path}")
    return scan_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Screen 119-bus scenarios with Perfect-Global plus rolling MPC-3 "
            "before running Proposed-H3."
        )
    )
    parser.add_argument("--candidate-scenarios", default="all")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--forecast-case", default="id_reliable")
    parser.add_argument("--time-limit-s", type=float, default=3600.0)
    parser.add_argument("--mpc-time-limit-s", type=float, default=300.0)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--seed-perfect-scan",
        default="formal/step10_119_perfect_global_scan_raw.csv",
        help=(
            "Optional local Perfect-Global scan raw used to avoid rerunning "
            "already certified candidates."
        ),
    )
    parser.add_argument("--no-seed-perfect-scan", action="store_true")
    parser.add_argument("--scan-output", default="formal/step10_119_operable_scan_raw.csv")
    parser.add_argument(
        "--selected-perfect-output",
        default="formal/step10_119_operable_perfect_selected20_raw.csv",
    )
    parser.add_argument(
        "--selected-mpc-output",
        default="formal/step10_119_operable_mpc3_selected20_raw.csv",
    )
    parser.add_argument(
        "--selected-json",
        default="formal/selected_119_operable_20.json",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_actual_case(args.actual_case)
    validate_forecast_case(args.forecast_case)

    candidate_scenarios = parse_csv_scenarios(
        args.candidate_scenarios,
        system="119",
        scenario_count=args.scenario_count,
    )
    scan_path = resolve_result_path(args.scan_output)
    selected_perfect_path = resolve_result_path(args.selected_perfect_output)
    selected_mpc_path = resolve_result_path(args.selected_mpc_output)
    json_path = resolve_result_path(args.selected_json)

    scan_rows: List[Dict[str, Any]]
    if args.resume:
        scan_rows = load_existing_rows(scan_path)
        print(f"Resume enabled. Loaded {len(scan_rows)} existing scan rows.")
    else:
        scan_rows = []
    if not args.no_seed_perfect_scan and args.seed_perfect_scan:
        scan_rows = seed_perfect_rows(
            scan_rows,
            seed_path=resolve_result_path(args.seed_perfect_scan),
            candidate_scenarios=candidate_scenarios,
            selected_json=args.selected_json,
        )

    row_map = rows_by_scenario_method(scan_rows)
    selected_ids = selected_scenario_ids(
        scan_rows,
        mip_gap_target=float(args.mip_gap),
        target_count=int(args.target_count),
    )
    print("Candidate scenarios:", candidate_scenarios)
    print("Existing selected scenarios:", selected_ids)
    print("Target count:", args.target_count)

    for order, scenario_id in enumerate(candidate_scenarios):
        if len(selected_ids) >= args.target_count:
            break

        perfect_key = (scenario_id, "Perfect-Global")
        mpc_key = (scenario_id, "MPC-3")

        perfect_row = row_map.get(perfect_key)
        if perfect_row is None:
            print(f"\n[Perfect scan] scenario={scenario_id}")
            perfect_row = run_one_perfect_global_119(
                scenario_id=scenario_id,
                actual_case=args.actual_case,
                quiet_solver=not args.verbose_solver,
                time_limit_s=args.time_limit_s,
                mip_gap=args.mip_gap,
            )
            scan_rows.append(
                mark_screening_row(
                    perfect_row,
                    candidate_order=order,
                    role="perfect_global",
                    selected_json=args.selected_json,
                    selected=False,
                )
            )
            row_map = rows_by_scenario_method(scan_rows)
            perfect_row = row_map[perfect_key]

        perfect_ok = is_selected_perfect_row(
            perfect_row,
            mip_gap_target=float(args.mip_gap),
        )
        if not perfect_ok:
            print(
                f"[NOT OPERABLE] scenario={scenario_id}, "
                f"perfect_completed={perfect_row.get('completed')}, "
                f"certified={perfect_row.get('certified_global')}, "
                f"status={perfect_row.get('gurobi_status')}"
            )
        else:
            mpc_row = row_map.get(mpc_key)
            if mpc_row is None:
                print(f"[MPC-3 scan] scenario={scenario_id}")
                mpc_row = run_one_scenario_119(
                    method="MPC-3",
                    solver_kind="pure_mpc_clean",
                    scenario_id=scenario_id,
                    forecast_case=args.forecast_case,
                    actual_case=args.actual_case,
                    horizon=3,
                    gamma_Q=None,
                    max_steps=None,
                    include_tail=True,
                    quiet_solver=not args.verbose_solver,
                    time_limit_s=args.mpc_time_limit_s,
                    mip_gap=args.mip_gap,
                )
                scan_rows.append(
                    mark_screening_row(
                        mpc_row,
                        candidate_order=order,
                        role="rolling_mpc3",
                        selected_json=args.selected_json,
                        selected=False,
                    )
                )
                row_map = rows_by_scenario_method(scan_rows)
                mpc_row = row_map[mpc_key]

            mpc_ok = is_selected_mpc3_row(mpc_row)
            if mpc_ok:
                print(f"[OPERABLE CANDIDATE] scenario={scenario_id}")
            else:
                print(
                    f"[NOT OPERABLE] scenario={scenario_id}, "
                    f"mpc_completed={mpc_row.get('completed')}, "
                    f"reason={mpc_row.get('incomplete_reason')}, "
                    f"steps={mpc_row.get('executed_steps')}"
                )

        selected_ids = selected_scenario_ids(
            scan_rows,
            mip_gap_target=float(args.mip_gap),
            target_count=int(args.target_count),
        )
        selected_set = set(selected_ids)
        scan_rows = [
            mark_screening_row(
                row,
                candidate_order=parse_int(row.get("operable_screening_candidate_order"), order),
                role=str(row.get("operable_screening_role", "")) or method_from_row(row),
                selected_json=args.selected_json,
                selected=scenario_id_from_row(row) in selected_set,
            )
            for row in scan_rows
        ]
        row_map = rows_by_scenario_method(scan_rows)

        write_screening_outputs(
            scan_path=scan_path,
            selected_perfect_path=selected_perfect_path,
            selected_mpc_path=selected_mpc_path,
            json_path=json_path,
            scan_rows=scan_rows,
            selected_ids=selected_ids,
            candidate_scenarios=candidate_scenarios,
            target_count=int(args.target_count),
            actual_case=args.actual_case,
            forecast_case=args.forecast_case,
            mip_gap=float(args.mip_gap),
            scan_output=args.scan_output,
            selected_perfect_output=args.selected_perfect_output,
            selected_mpc_output=args.selected_mpc_output,
        )
        print(f"Selected count: {len(selected_ids)}/{args.target_count}")

    selected_ids = selected_scenario_ids(
        scan_rows,
        mip_gap_target=float(args.mip_gap),
        target_count=int(args.target_count),
    )
    write_screening_outputs(
        scan_path=scan_path,
        selected_perfect_path=selected_perfect_path,
        selected_mpc_path=selected_mpc_path,
        json_path=json_path,
        scan_rows=scan_rows,
        selected_ids=selected_ids,
        candidate_scenarios=candidate_scenarios,
        target_count=int(args.target_count),
        actual_case=args.actual_case,
        forecast_case=args.forecast_case,
        mip_gap=float(args.mip_gap),
        scan_output=args.scan_output,
        selected_perfect_output=args.selected_perfect_output,
        selected_mpc_output=args.selected_mpc_output,
    )

    print("\nFinal selected scenarios:", selected_ids)
    print(f"Selected count: {len(selected_ids)}/{args.target_count}")

    if len(selected_ids) < args.target_count:
        raise SystemExit(
            f"Only found {len(selected_ids)} operable 119-bus scenarios. "
            f"Need {args.target_count}. Scan more candidate scenarios or reduce target-count."
        )


if __name__ == "__main__":
    main()
