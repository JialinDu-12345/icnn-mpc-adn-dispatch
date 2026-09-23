from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


DEFAULT_INPUTS = [
    "formal/step7_33_oracle_mpc_raw.csv",
    "formal/step4_33_main_model_based_raw.csv",
    "formal/step7_33_policy_baselines_raw.csv",
]

EXTRA_COLUMNS = [
    "source_raw",
    "oracle_method",
    "oracle_cost",
    "gap_to_oracle_percent",
]


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    return float(text)


def read_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def ordered_fieldnames(rows: List[Dict[str, object]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    for key in EXTRA_COLUMNS:
        if key not in fieldnames:
            fieldnames.append(key)
    return fieldnames


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=ordered_fieldnames(rows), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def row_completed(row: Dict[str, str]) -> bool:
    if row.get("completed") not in (None, ""):
        return parse_bool(row.get("completed"))
    return (
        int(float(row.get("executed_steps") or 0)) == 48
        and int(float(row.get("solver_fail_count") or 0)) == 0
    )


def scenario_key(row: Dict[str, str]) -> Tuple[str, str, str]:
    return (
        str(row.get("system", "33")),
        str(row.get("actual_case", "id_actual")),
        str(row.get("scenario_id", "")),
    )


def build_oracle_lookup(
    rows: List[Dict[str, str]],
    oracle_method: str,
    allow_incomplete_oracle: bool,
) -> Dict[Tuple[str, str, str], float]:
    lookup: Dict[Tuple[str, str, str], float] = {}
    for row in rows:
        if str(row.get("method", "")) != oracle_method:
            continue
        if not allow_incomplete_oracle and not row_completed(row):
            continue
        key = scenario_key(row)
        cost = parse_float(row.get("cost"))
        if not key[2] or not math.isfinite(cost):
            continue
        if key in lookup and not math.isclose(lookup[key], cost, rel_tol=1e-9, abs_tol=1e-9):
            raise RuntimeError(f"Duplicate oracle cost for {key}: {lookup[key]} vs {cost}")
        lookup[key] = cost
    return lookup


def add_oracle_gap(
    rows: List[Dict[str, str]],
    source_raw: str,
    oracle_lookup: Dict[Tuple[str, str, str], float],
    oracle_method: str,
    completed_only: bool,
    strict: bool,
) -> List[Dict[str, object]]:
    output_rows: List[Dict[str, object]] = []
    missing_oracle: List[Tuple[str, str, str]] = []

    for row in rows:
        if completed_only and not row_completed(row):
            continue

        normalized: Dict[str, object] = dict(row)
        normalized["source_raw"] = source_raw
        normalized["oracle_method"] = oracle_method

        key = scenario_key(row)
        cost = parse_float(row.get("cost"))
        oracle_cost = oracle_lookup.get(key)
        if oracle_cost is None:
            normalized["oracle_cost"] = ""
            normalized["gap_to_oracle_percent"] = ""
            if key[2] and math.isfinite(cost):
                missing_oracle.append(key)
        else:
            normalized["oracle_cost"] = oracle_cost
            if math.isfinite(cost) and oracle_cost != 0.0:
                normalized["gap_to_oracle_percent"] = (cost - oracle_cost) / oracle_cost * 100.0
            else:
                normalized["gap_to_oracle_percent"] = ""

        output_rows.append(normalized)

    if strict and missing_oracle:
        preview = missing_oracle[:10]
        raise RuntimeError(
            f"Missing {oracle_method} oracle costs for {len(missing_oracle)} rows. "
            f"First missing keys: {preview}"
        )

    return output_rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Add matched Oracle gap to 33-bus raw results.")
    parser.add_argument("--inputs", default=",".join(DEFAULT_INPUTS))
    parser.add_argument("--oracle-raw", default="formal/step7_33_oracle_mpc_raw.csv")
    parser.add_argument("--oracle-method", default="Oracle-H7")
    parser.add_argument("--output", default="formal/step8_33_main_table_with_gap_raw.csv")
    parser.add_argument("--completed-only", action="store_true")
    parser.add_argument(
        "--allow-incomplete-oracle",
        action="store_true",
        help="Use incomplete Oracle rows too. Intended only for one-step smoke checks.",
    )
    parser.add_argument("--strict", action="store_true")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    oracle_path = RESULT_DIR / args.oracle_raw
    if not oracle_path.exists():
        raise FileNotFoundError(f"Missing oracle raw file: {oracle_path}")

    oracle_rows = read_rows(oracle_path)
    oracle_lookup = build_oracle_lookup(
        oracle_rows,
        oracle_method=args.oracle_method,
        allow_incomplete_oracle=args.allow_incomplete_oracle,
    )
    if not oracle_lookup:
        raise RuntimeError(
            f"No usable oracle rows found for method {args.oracle_method}. "
            "Check --oracle-method or pass --allow-incomplete-oracle for smoke files."
        )

    output_rows: List[Dict[str, object]] = []
    for input_name in parse_csv_list(args.inputs):
        input_path = RESULT_DIR / input_name
        if not input_path.exists():
            message = f"Missing input raw file: {input_path}"
            if args.strict:
                raise FileNotFoundError(message)
            print("[Skipped]", message)
            continue
        rows = read_rows(input_path)
        output_rows.extend(
            add_oracle_gap(
                rows,
                source_raw=input_name,
                oracle_lookup=oracle_lookup,
                oracle_method=args.oracle_method,
                completed_only=args.completed_only,
                strict=args.strict,
            )
        )

    output_path = RESULT_DIR / args.output
    write_rows(output_path, output_rows)
    print(f"Oracle rows used: {len(oracle_lookup)}")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
