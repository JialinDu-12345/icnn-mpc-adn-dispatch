from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RAW_RESULT_COLUMNS, validate_actual_case              
from experiments_v2.global_optimum.run_119_perfect_global import (              
    run_one_perfect_global_119,
)
from experiments_v2.utils.scenario_selection import (              
    parse_csv_scenarios,
    resolve_result_path,
)


SELECTION_RULE = "perfect_global_feasible_certified_subset"
SCREENING_RULE = "completed_and_certified_full_day_hard_opf"


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    try:
        return float(text)
    except ValueError:
        return default


def parse_int(value: object, default: int = 0) -> int:
    parsed = parse_float(value)
    if not math.isfinite(parsed):
        return default
    return int(parsed)


def is_selected_perfect_row(row: Dict[str, Any], mip_gap_target: float) -> bool:
    completed = parse_bool(row.get("completed"))
    certified = parse_bool(row.get("certified_global"))
    fail_count = parse_int(row.get("solver_fail_count"), default=0)
    bin_vars = parse_float(row.get("mean_num_bin_vars"))
    mip_gap = parse_float(row.get("mean_mip_gap"))

    if not completed or not certified:
        return False
    if fail_count != 0:
        return False
    if math.isfinite(bin_vars) and abs(bin_vars) > 1e-9:
        return False
    if math.isfinite(mip_gap) and mip_gap > mip_gap_target + 1e-9:
        return False
    return True


def load_existing_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def fieldnames_for_rows(rows: List[Dict[str, Any]]) -> List[str]:
    fieldnames = list(RAW_RESULT_COLUMNS)
    known = set(fieldnames)
    for row in rows:
        for key in row:
            if key not in known:
                fieldnames.append(key)
                known.add(key)
    return fieldnames


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames_for_rows(rows),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def scenario_id_from_row(row: Dict[str, Any]) -> int:
    return parse_int(row.get("scenario_id"), default=10**9)


def sort_rows_by_scenario(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(rows, key=scenario_id_from_row)


def select_rows(
    rows: List[Dict[str, Any]],
    *,
    mip_gap_target: float,
    target_count: int,
    selected_json: str,
) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    seen = set()
    for row in sort_rows_by_scenario(rows):
        scenario_id = scenario_id_from_row(row)
        if scenario_id in seen:
            continue
        if not is_selected_perfect_row(row, mip_gap_target=mip_gap_target):
            continue
        out = dict(row)
        out["scenario_selection_file"] = selected_json
        out["scenario_selection_rule"] = SELECTION_RULE
        selected.append(out)
        seen.add(scenario_id)
        if len(selected) >= target_count:
            break
    return selected


def save_selection_json(
    *,
    path: Path,
    selected_scenarios: List[int],
    candidate_scenarios: List[int],
    scanned_scenarios: List[int],
    target_count: int,
    scan_output: str,
    selected_output: str,
    actual_case: str,
    mip_gap: float,
) -> None:
    payload = {
        "system": "119",
        "selection_rule": SELECTION_RULE,
        "screening_rule": SCREENING_RULE,
        "actual_case": actual_case,
        "target_count": int(target_count),
        "selected_count": len(selected_scenarios),
        "selected_scenarios": selected_scenarios,
        "candidate_scenarios": candidate_scenarios,
        "scanned_scenarios": scanned_scenarios,
        "n_candidate_scenarios": len(candidate_scenarios),
        "n_candidates_scanned": len(scanned_scenarios),
        "mip_gap_target": float(mip_gap),
        "scan_output": scan_output,
        "selected_output": selected_output,
        "note": (
            "Scenarios are selected only by feasibility/certification of the "
            "full-day Perfect-Global benchmark, before evaluating MPC, policy, "
            "or Proposed-H3 methods."
        ),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("Saved selected scenario JSON:", path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Screen 119-bus test scenarios using Perfect-Global feasibility."
    )
    parser.add_argument("--candidate-scenarios", default="all")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--time-limit-s", type=float, default=3600.0)
    parser.add_argument("--mip-gap", type=float, default=1e-3)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scan-output", default="formal/step10_119_perfect_global_scan_raw.csv")
    parser.add_argument(
        "--selected-output",
        default="formal/step10_119_perfect_global_selected20_raw.csv",
    )
    parser.add_argument(
        "--selected-json",
        default="formal/selected_119_perfect_feasible_20.json",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_actual_case(args.actual_case)

    candidate_scenarios = parse_csv_scenarios(
        args.candidate_scenarios,
        system="119",
        scenario_count=args.scenario_count,
    )
    scan_path = resolve_result_path(args.scan_output)
    selected_path = resolve_result_path(args.selected_output)
    json_path = resolve_result_path(args.selected_json)

    scan_rows: List[Dict[str, Any]]
    if args.resume:
        scan_rows = load_existing_rows(scan_path)
        print(f"Resume enabled. Loaded {len(scan_rows)} existing scan rows.")
    else:
        scan_rows = []

    scanned_ids = {scenario_id_from_row(row) for row in scan_rows}
    selected_rows = select_rows(
        scan_rows,
        mip_gap_target=float(args.mip_gap),
        target_count=int(args.target_count),
        selected_json=args.selected_json,
    )
    selected_ids = [scenario_id_from_row(row) for row in selected_rows]

    print("Candidate scenarios:", candidate_scenarios)
    print("Existing selected scenarios:", selected_ids)
    print("Target count:", args.target_count)

    for order, scenario_id in enumerate(candidate_scenarios):
        if len(selected_ids) >= args.target_count:
            break

        if args.resume and scenario_id in scanned_ids:
            print(f"[SKIP] scenario={scenario_id} already scanned.")
            continue

        print(f"\n[Perfect scan] scenario={scenario_id}")
        row = run_one_perfect_global_119(
            scenario_id=scenario_id,
            actual_case=args.actual_case,
            quiet_solver=not args.verbose_solver,
            time_limit_s=args.time_limit_s,
            mip_gap=args.mip_gap,
        )
        row["perfect_screening_target_count"] = args.target_count
        row["perfect_screening_candidate_order"] = order
        row["perfect_screening_rule"] = SCREENING_RULE
        row["scenario_selection_file"] = ""
        row["scenario_selection_rule"] = "perfect_global_screening_candidate"

        if is_selected_perfect_row(row, mip_gap_target=float(args.mip_gap)):
            row["scenario_selection_file"] = args.selected_json
            row["scenario_selection_rule"] = SELECTION_RULE
            print(f"[SELECTED] scenario={scenario_id}")
        else:
            print(
                f"[NOT SELECTED] scenario={scenario_id}, "
                f"completed={row.get('completed')}, "
                f"certified={row.get('certified_global')}, "
                f"status={row.get('gurobi_status')}"
            )

        scan_rows.append(row)
        scanned_ids.add(scenario_id)
        selected_rows = select_rows(
            scan_rows,
            mip_gap_target=float(args.mip_gap),
            target_count=int(args.target_count),
            selected_json=args.selected_json,
        )
        selected_ids = [scenario_id_from_row(item) for item in selected_rows]

        write_rows(scan_path, sort_rows_by_scenario(scan_rows))
        write_rows(selected_path, selected_rows)
        save_selection_json(
            path=json_path,
            selected_scenarios=selected_ids,
            candidate_scenarios=candidate_scenarios,
            scanned_scenarios=sorted(scanned_ids),
            target_count=int(args.target_count),
            scan_output=args.scan_output,
            selected_output=args.selected_output,
            actual_case=args.actual_case,
            mip_gap=float(args.mip_gap),
        )
        print(f"Selected count: {len(selected_ids)}/{args.target_count}")

    selected_rows = select_rows(
        scan_rows,
        mip_gap_target=float(args.mip_gap),
        target_count=int(args.target_count),
        selected_json=args.selected_json,
    )
    selected_ids = [scenario_id_from_row(row) for row in selected_rows]
    write_rows(scan_path, sort_rows_by_scenario(scan_rows))
    write_rows(selected_path, selected_rows)
    save_selection_json(
        path=json_path,
        selected_scenarios=selected_ids,
        candidate_scenarios=candidate_scenarios,
        scanned_scenarios=sorted(scanned_ids),
        target_count=int(args.target_count),
        scan_output=args.scan_output,
        selected_output=args.selected_output,
        actual_case=args.actual_case,
        mip_gap=float(args.mip_gap),
    )

    print("\nFinal selected scenarios:", selected_ids)
    print(f"Selected count: {len(selected_ids)}/{args.target_count}")

    if len(selected_ids) < args.target_count:
        raise SystemExit(
            f"Only found {len(selected_ids)} feasible/certified Perfect-Global scenarios. "
            f"Need {args.target_count}. Scan more candidate scenarios or reduce target-count."
        )


if __name__ == "__main__":
    main()
