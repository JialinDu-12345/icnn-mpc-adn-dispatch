from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR, SPLITS, validate_actual_case, validate_forecast_case              
from experiments_v2.global_optimum.run_33_perfect_global import (              
    run_one_perfect_global_33,
)
from experiments_v2.run_33_mpc import run_one_scenario_33              


DEFAULT_RELAXED_RAMP_VALUE = 0.8


@dataclass(frozen=True)
class DiagnosticCase:
    case_id: str
    display_name: str
    modified_property: str
    price_variation: str
    time_coupling: str
    flat_price: bool = False
    relaxed_ramp: bool = False


DIAGNOSTIC_CASES: Dict[str, DiagnosticCase] = {
    "original": DiagnosticCase(
        case_id="original",
        display_name="Original",
        modified_property="None",
        price_variation="High",
        time_coupling="Strong",
    ),
    "flat_price": DiagnosticCase(
        case_id="flat_price",
        display_name="Flat price",
        modified_property="Remove price variation",
        price_variation="Removed",
        time_coupling="Strong",
        flat_price=True,
    ),
    "relaxed_ramp": DiagnosticCase(
        case_id="relaxed_ramp",
        display_name="Relaxed ramp",
        modified_property="Weaken DG inter-temporal coupling",
        price_variation="High",
        time_coupling="Weak DG ramp",
        relaxed_ramp=True,
    ),
    "flat_price_relaxed_ramp": DiagnosticCase(
        case_id="flat_price_relaxed_ramp",
        display_name="Flat price + relaxed ramp",
        modified_property="Remove dominant long-term effects",
        price_variation="Removed",
        time_coupling="Weak DG ramp",
        flat_price=True,
        relaxed_ramp=True,
    ),
}

CASE_ALIASES = {
    "all": list(DIAGNOSTIC_CASES),
    "flat": ["flat_price"],
    "relaxed": ["relaxed_ramp"],
    "easy": ["flat_price_relaxed_ramp"],
    "flat_relaxed": ["flat_price_relaxed_ramp"],
    "flat+ramp": ["flat_price_relaxed_ramp"],
}

METHOD_ALIASES = {
    "mpc3": "MPC-3",
    "mpc-3": "MPC-3",
    "perfect": "Perfect-Global",
    "perfect-global": "Perfect-Global",
    "perfect_global": "Perfect-Global",
}

METHOD_ORDER = ["MPC-3", "Perfect-Global"]


def parse_csv_list(raw: Optional[str]) -> List[str]:
    if raw is None:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_cases(raw: str) -> List[DiagnosticCase]:
    selected: List[str] = []
    seen = set()
    for item in parse_csv_list(raw):
        lowered = item.lower()
        case_ids = CASE_ALIASES.get(lowered, [lowered])
        for case_id in case_ids:
            if case_id not in DIAGNOSTIC_CASES:
                raise ValueError(
                    f"Unknown diagnostic case '{item}'. "
                    f"Available cases: {list(DIAGNOSTIC_CASES)}"
                )
            if case_id in seen:
                continue
            selected.append(case_id)
            seen.add(case_id)
    return [DIAGNOSTIC_CASES[case_id] for case_id in selected]


def parse_methods(raw: str) -> List[str]:
    methods: List[str] = []
    seen = set()
    for item in parse_csv_list(raw):
        method = METHOD_ALIASES.get(item.lower(), item)
        if method not in METHOD_ORDER:
            raise ValueError(f"Unknown method '{item}'. Available methods: {METHOD_ORDER}")
        if method in seen:
            continue
        methods.append(method)
        seen.add(method)
    return methods


def parse_scenarios(raw: str, scenario_count: Optional[int]) -> List[int]:
    normalized = raw.strip().lower()
    if normalized == "all":
        scenario_ids = list(SPLITS["33"]["test_id"])
    elif normalized in {"val", "validation"}:
        scenario_ids = list(SPLITS["33"]["val"])
    else:
        scenario_ids = [int(item) for item in parse_csv_list(raw)]

    if scenario_count is not None:
        scenario_ids = scenario_ids[:scenario_count]
    return scenario_ids


def finite_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text:
        return default
    try:
        numeric = float(text)
    except (TypeError, ValueError):
        return default
    return numeric if math.isfinite(numeric) else default


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def completed(row: Dict[str, Any]) -> bool:
    if "completed" in row and str(row.get("completed")).strip() != "":
        return parse_bool(row.get("completed"))
    return int(finite_float(row.get("solver_fail_count"), 1.0)) == 0


def is_mpc_method(method: object) -> bool:
    return str(method).strip().startswith("MPC-")


def find_mpc_row(method_rows: Dict[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for method_name, row in method_rows.items():
        if is_mpc_method(method_name):
            return row
    return None


def ci95(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return math.nan
    return 1.96 * stdev(clean) / math.sqrt(len(clean))


def mean_or_nan(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if not clean:
        return math.nan
    return mean(clean)


def std_or_nan(values: Iterable[float]) -> float:
    clean = [float(value) for value in values if math.isfinite(float(value))]
    if len(clean) < 2:
        return math.nan
    return stdev(clean)


def make_solver_customizer(
    case: DiagnosticCase,
    relaxed_ramp_value: Optional[float],
) -> Callable[[Any], None]:
    def customize(solver: Any) -> None:
        prices = np.asarray(getattr(solver, "Prices"), dtype=float)
        if case.flat_price:
            solver.Prices = np.full(prices.shape, float(np.mean(prices)), dtype=float)

        if case.relaxed_ramp:
            if relaxed_ramp_value is None:
                solver.gen_ramp = DEFAULT_RELAXED_RAMP_VALUE
            else:
                solver.gen_ramp = float(relaxed_ramp_value)

    return customize


def decorate_row(
    row: Dict[str, Any],
    *,
    case: DiagnosticCase,
    relaxed_ramp_value: Optional[float],
) -> Dict[str, Any]:
    row = dict(row)
    effective_relaxed_ramp_value = relaxed_ramp_value if case.relaxed_ramp else None
    relaxed_ramp_setting = ""
    if case.relaxed_ramp:
        relaxed_ramp_setting = (
            str(effective_relaxed_ramp_value)
            if effective_relaxed_ramp_value is not None
            else str(DEFAULT_RELAXED_RAMP_VALUE)
        )
    row.update(
        {
            "diagnostic_case": case.case_id,
            "diagnostic_case_label": case.display_name,
            "modified_property": case.modified_property,
            "price_variation": case.price_variation,
            "time_coupling": case.time_coupling,
            "flat_price": case.flat_price,
            "relaxed_ramp": case.relaxed_ramp,
            "relaxed_ramp_value": effective_relaxed_ramp_value,
            "relaxed_ramp_setting": relaxed_ramp_setting,
            "experiment_tag": "mpc_diagnostic",
        }
    )
    return row


def add_pair_gap(rows: List[Dict[str, Any]]) -> None:
    by_key: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (row.get("diagnostic_case"), int(row.get("scenario_id")))
        by_key[key][str(row.get("method"))] = row

    for pair in by_key.values():
        mpc = find_mpc_row(pair)
        perfect = pair.get("Perfect-Global")
        if mpc is None or perfect is None:
            continue
        mpc_cost = finite_float(mpc.get("cost"))
        perfect_cost = finite_float(perfect.get("cost"))
        if not math.isfinite(mpc_cost) or not math.isfinite(perfect_cost) or perfect_cost == 0:
            continue
        gap = 100.0 * (mpc_cost - perfect_cost) / perfect_cost
        mpc["perfect_cost"] = perfect_cost
        mpc["gap_percent"] = gap
        mpc["gap_to_perfect_percent"] = gap
        perfect["perfect_cost"] = perfect_cost
        perfect["gap_percent"] = 0.0
        perfect["gap_to_perfect_percent"] = 0.0


def build_summary_rows(rows: List[Dict[str, Any]], cases: List[DiagnosticCase]) -> List[Dict[str, Any]]:
    by_case_scenario: Dict[tuple, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("diagnostic_case")), int(row.get("scenario_id")))
        by_case_scenario[key][str(row.get("method"))] = row

    summary_rows: List[Dict[str, Any]] = []
    for case in cases:
        pair_rows = []
        mpc_attempted_rows = []
        mpc_only_rows = []
        perfect_attempted_rows = []
        perfect_only_rows = []
        for (case_id, _scenario_id), methods in by_case_scenario.items():
            if case_id != case.case_id:
                continue
            mpc = find_mpc_row(methods)
            perfect = methods.get("Perfect-Global")
            if mpc is not None:
                mpc_attempted_rows.append(mpc)
                if completed(mpc):
                    mpc_only_rows.append(mpc)
            if perfect is not None:
                perfect_attempted_rows.append(perfect)
                if completed(perfect):
                    perfect_only_rows.append(perfect)
            if (
                mpc is not None
                and perfect is not None
                and completed(mpc)
                and completed(perfect)
                and math.isfinite(finite_float(mpc.get("cost")))
                and math.isfinite(finite_float(perfect.get("cost")))
            ):
                pair_rows.append((mpc, perfect))

        mpc_costs = [finite_float(row.get("cost")) for row in mpc_only_rows]
        perfect_costs = [finite_float(row.get("cost")) for row in perfect_only_rows]
        gaps = [finite_float(mpc.get("gap_percent")) for mpc, _perfect in pair_rows]
        mpc_step_times = [finite_float(row.get("avg_step_time_s")) for row in mpc_only_rows]
        perfect_times = [finite_float(row.get("solve_time_s")) for row in perfect_only_rows]
        representative_mpc = mpc_attempted_rows[0] if mpc_attempted_rows else {}
        mpc_method = str(representative_mpc.get("method", "MPC"))
        mpc_horizon = row_int(representative_mpc, "horizon", default=-1)
        forecast_case = str(representative_mpc.get("forecast_case", ""))
        mpc_timeout_total = sum(
            max(row_int(row, "timeout_count", default=0), 0) for row in mpc_attempted_rows
        )
        mpc_solver_fail_total = sum(
            max(row_int(row, "solver_fail_count", default=0), 0)
            for row in mpc_attempted_rows
        )

        summary_rows.append(
            {
                "diagnostic_case": case.case_id,
                "case": case.display_name,
                "modified_property": case.modified_property,
                "price_variation": case.price_variation,
                "time_coupling": case.time_coupling,
                "mpc_method": mpc_method,
                "mpc_horizon": mpc_horizon if mpc_horizon >= 0 else "",
                "forecast_case": forecast_case,
                "n_mpc_attempted": len(mpc_attempted_rows),
                "n_mpc_completed": len(mpc_only_rows),
                "n_mpc3_completed": len(mpc_only_rows),
                "n_mpc_failed": len(mpc_attempted_rows) - len(mpc_only_rows),
                "mpc_completion_rate_percent": (
                    100.0 * len(mpc_only_rows) / len(mpc_attempted_rows)
                    if mpc_attempted_rows
                    else math.nan
                ),
                "mpc_solver_fail_count_total": mpc_solver_fail_total,
                "mpc_timeout_count_total": mpc_timeout_total,
                "n_perfect_attempted": len(perfect_attempted_rows),
                "n_perfect_completed": len(perfect_only_rows),
                "n_matched_pairs": len(pair_rows),
                "mean_mpc_cost": mean_or_nan(mpc_costs),
                "std_mpc_cost": std_or_nan(mpc_costs),
                "ci95_mpc_cost": ci95(mpc_costs),
                "mean_mpc3_cost": mean_or_nan(mpc_costs),
                "std_mpc3_cost": std_or_nan(mpc_costs),
                "ci95_mpc3_cost": ci95(mpc_costs),
                "mean_perfect_cost": mean_or_nan(perfect_costs),
                "std_perfect_cost": std_or_nan(perfect_costs),
                "ci95_perfect_cost": ci95(perfect_costs),
                "mean_gap_percent": mean_or_nan(gaps),
                "std_gap_percent": std_or_nan(gaps),
                "ci95_gap_percent": ci95(gaps),
                "mean_mpc_avg_step_time_s": mean_or_nan(mpc_step_times),
                "std_mpc_avg_step_time_s": std_or_nan(mpc_step_times),
                "ci95_mpc_avg_step_time_s": ci95(mpc_step_times),
                "mean_mpc3_avg_step_time_s": mean_or_nan(mpc_step_times),
                "std_mpc3_avg_step_time_s": std_or_nan(mpc_step_times),
                "ci95_mpc3_avg_step_time_s": ci95(mpc_step_times),
                "mean_perfect_solve_time_s": mean_or_nan(perfect_times),
                "std_perfect_solve_time_s": std_or_nan(perfect_times),
                "ci95_perfect_solve_time_s": ci95(perfect_times),
            }
        )
    return summary_rows


def collect_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return fieldnames


def write_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"[Skip] No rows to write for {path}.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = collect_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("Saved:", path)


def resolve_result_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else RESULT_DIR / path


def with_case_suffix(path_like: str, case_id: str) -> Path:
    path = resolve_result_path(path_like)
    return path.with_name(f"{path.stem}_{case_id}{path.suffix}")


def save_case_outputs(
    *,
    case: DiagnosticCase,
    case_rows: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    if not case_rows:
        print(f"[Warning] No rows for case={case.case_id}; skip case-level outputs.")
        return

    add_pair_gap(case_rows)

    raw_path = with_case_suffix(args.output, case.case_id)
    summary_path = with_case_suffix(args.summary_output, case.case_id)
    table_dir = resolve_result_path(args.table_output_dir) / case.case_id

    write_rows(raw_path, case_rows)

    summary_rows = build_summary_rows(case_rows, [case])
    write_rows(summary_path, summary_rows)
    write_paper_table(summary_rows, table_dir)

    print(f"[Case saved] {case.case_id}")


def read_csv_rows(path_like: str) -> List[Dict[str, Any]]:
    path = resolve_result_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Cannot find cached result file: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def row_int(row: Dict[str, Any], key: str, default: int = -1) -> int:
    try:
        return int(float(str(row.get(key, "")).strip()))
    except (TypeError, ValueError):
        return default


def load_original_cached_rows(
    *,
    args: argparse.Namespace,
    scenario_ids: List[int],
    methods: List[str],
    mpc_method_name: str,
) -> List[Dict[str, Any]]:
    case = DIAGNOSTIC_CASES["original"]
    scenario_set = {int(sid) for sid in scenario_ids}
    cached_rows: List[Dict[str, Any]] = []

    if "MPC-3" in methods:
        for row in read_csv_rows(args.original_mpc3_input):
            if str(row.get("method")) != mpc_method_name:
                continue
            if row_int(row, "scenario_id") not in scenario_set:
                continue
            if str(row.get("actual_case")) != str(args.actual_case):
                continue
            if str(row.get("forecast_case")) != str(args.forecast_case):
                continue
            if row_int(row, "horizon", default=0) != int(args.mpc_horizon):
                continue

            row = decorate_row(row, case=case, relaxed_ramp_value=None)
            row["source_cache"] = args.original_mpc3_input
            cached_rows.append(row)

    if "Perfect-Global" in methods:
        for row in read_csv_rows(args.original_perfect_input):
            if str(row.get("method")) != "Perfect-Global":
                continue
            if row_int(row, "scenario_id") not in scenario_set:
                continue
            if str(row.get("actual_case")) != str(args.actual_case):
                continue

            row = decorate_row(row, case=case, relaxed_ramp_value=None)
            row["source_cache"] = args.original_perfect_input
            cached_rows.append(row)

    got = {
        (row_int(row, "scenario_id"), str(row.get("method")))
        for row in cached_rows
    }
    expected = {
        (int(sid), mpc_method_name if method == "MPC-3" else method)
        for sid in scenario_ids
        for method in methods
    }
    missing = sorted(expected - got)
    if missing:
        raise RuntimeError(
            "Missing cached original results for: "
            + ", ".join(f"scenario={sid}, method={method}" for sid, method in missing)
        )

    add_pair_gap(cached_rows)
    return cached_rows


def load_perfect_cache_index(path_like: str) -> Dict[tuple, Dict[str, Any]]:
    rows = read_csv_rows(path_like)
    index: Dict[tuple, Dict[str, Any]] = {}

    for row in rows:
        if str(row.get("method")) != "Perfect-Global":
            continue
        if not completed(row):
            continue

        case_id = str(row.get("diagnostic_case", "")).strip()
        scenario_id = row_int(row, "scenario_id")
        actual_case = str(row.get("actual_case", "id_actual"))
        if not case_id or scenario_id < 0:
            continue
        index[(case_id, scenario_id, actual_case)] = dict(row)

    if not index:
        raise RuntimeError(
            f"No completed Perfect-Global diagnostic rows found in {resolve_result_path(path_like)}"
        )
    return index


def format_number(value: object, digits: int = 2) -> str:
    numeric = finite_float(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [
        max(len(headers[idx]), *(len(row[idx]) for row in rows)) if rows else len(headers[idx])
        for idx in range(len(headers))
    ]
    header_line = "| " + " | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def latex_escape(text: object) -> str:
    value = "" if text is None else str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    return value


def latex_table(headers: List[str], rows: List[List[str]]) -> str:
    column_spec = "l" + "c" * (len(headers) - 1)
    lines = [
        r"\begin{tabular}{" + column_spec + r"}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_paper_table(summary_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    mpc_labels = {
        str(row.get("mpc_method", "MPC"))
        for row in summary_rows
        if str(row.get("mpc_method", "")).strip()
    }
    mpc_label = next(iter(mpc_labels)) if len(mpc_labels) == 1 else "MPC"
    headers = [
        "Case",
        "Modified property",
        f"{mpc_label} cost mean ($)",
        f"{mpc_label} cost 95% CI ($)",
        "Perfect-Global cost mean ($)",
        "Perfect-Global cost 95% CI ($)",
        "Gap mean (%)",
        "Gap 95% CI (%)",
        f"{mpc_label} step time mean (s)",
        f"{mpc_label} step time 95% CI (s)",
        "Perfect-Global solve time mean (s)",
        "Perfect-Global solve time 95% CI (s)",
    ]
    table_rows = []
    for row in summary_rows:
        table_rows.append(
            [
                str(row["case"]),
                str(row["modified_property"]),
                format_number(row["mean_mpc_cost"], 2),
                format_number(row["ci95_mpc_cost"], 2),
                format_number(row["mean_perfect_cost"], 2),
                format_number(row["ci95_perfect_cost"], 2),
                format_number(row["mean_gap_percent"], 2),
                format_number(row["ci95_gap_percent"], 2),
                format_number(row["mean_mpc_avg_step_time_s"], 3),
                format_number(row["ci95_mpc_avg_step_time_s"], 3),
                format_number(row["mean_perfect_solve_time_s"], 3),
                format_number(row["ci95_perfect_solve_time_s"], 3),
            ]
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "table_33_mpc_diagnostic.csv"
    md_path = output_dir / "table_33_mpc_diagnostic.md"
    tex_path = output_dir / "table_33_mpc_diagnostic.tex"

    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table_rows)
    md_path.write_text(markdown_table(headers, table_rows), encoding="utf-8")
    tex_path.write_text(latex_table(headers, table_rows), encoding="utf-8")

    print("Saved:", csv_path)
    print("Saved:", md_path)
    print("Saved:", tex_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the 33-bus MPC failure and sufficiency diagnostic: "
            "finite-horizon MPC versus Perfect-Global under controlled variants."
        )
    )
    parser.add_argument("--scenarios", default="all", help="Comma-separated IDs, 'val', or 'all'.")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument(
        "--cases",
        default="all",
        help=(
            "Comma-separated cases: original, flat_price, relaxed_ramp, "
            "flat_price_relaxed_ramp, or all."
        ),
    )
    parser.add_argument(
        "--methods",
        default="MPC-3,Perfect-Global",
        help=(
            "Comma-separated methods. Use MPC-3 to enable MPC; its actual horizon and "
            "output label are controlled by --mpc-horizon."
        ),
    )
    parser.add_argument("--forecast-case", default="perfect")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument(
        "--mpc-horizon",
        type=int,
        default=3,
        help="MPC prediction horizon used in the diagnostic experiment.",
    )
    parser.add_argument(
        "--relaxed-ramp-value",
        type=float,
        default=DEFAULT_RELAXED_RAMP_VALUE,
        help=(
            "Optional raw solver gen_ramp value for relaxed-ramp cases. "
            f"Default is {DEFAULT_RELAXED_RAMP_VALUE}, i.e., twice the original "
            "half-hour ramp allowance when the copied solver applies gen_ramp / 2."
        ),
    )
    parser.set_defaults(reuse_original=False)
    parser.add_argument(
        "--reuse-original",
        dest="reuse_original",
        action="store_true",
        help=(
            "Reuse existing original-case MPC-3 and Perfect-Global results instead of "
            "rerunning them."
        ),
    )
    parser.add_argument(
        "--rerun-original",
        dest="reuse_original",
        action="store_false",
        help=(
            "Rerun the original case instead of reading the cached formal results. "
            "This is the default for direct PyCharm runs."
        ),
    )
    parser.add_argument(
        "--original-mpc3-input",
        default="formal/step4_33_main_model_based_raw.csv",
        help="Existing raw CSV containing original-case MPC-3 results.",
    )
    parser.add_argument(
        "--original-perfect-input",
        default="formal/step7b_33_perfect_global_raw.csv",
        help="Existing raw CSV containing original-case Perfect-Global results.",
    )
    parser.set_defaults(reuse_perfect_global=True)
    parser.add_argument(
        "--reuse-perfect-global",
        dest="reuse_perfect_global",
        action="store_true",
        help=(
            "Reuse cached Perfect-Global diagnostic rows. Perfect-Global is independent "
            "of the rolling forecast case. This is the default for direct PyCharm runs."
        ),
    )
    parser.add_argument(
        "--rerun-perfect-global",
        dest="reuse_perfect_global",
        action="store_false",
        help="Rerun Perfect-Global instead of reading the diagnostic cache.",
    )
    parser.add_argument(
        "--perfect-cache-input",
        default="formal/step12_33_mpc_diagnostic_raw.csv",
        help="Cached diagnostic raw CSV containing Perfect-Global rows for all cases.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Debug only; limits MPC steps.")
    parser.add_argument("--no-tail", action="store_true", help="Debug only; disables MPC tail solves.")
    parser.add_argument("--mpc-time-limit-s", type=float, default=300)
    parser.add_argument("--perfect-time-limit-s", type=float, default=1800)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument("--verbose-solver", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--output",
        default="formal/step12_33_mpc_diagnostic_perfect_forecast_H3_raw.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="formal/step12_33_mpc_diagnostic_perfect_forecast_H3_summary.csv",
    )
    parser.add_argument(
        "--table-output-dir",
        default="paper_tables/33/mpc_diagnostic_perfect_forecast_H3",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.mpc_horizon < 1:
        parser.error("--mpc-horizon must be at least 1.")
    validate_forecast_case(args.forecast_case)
    validate_actual_case(args.actual_case)
    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    cases = parse_cases(args.cases)
    methods = parse_methods(args.methods)
    mpc_method_name = f"MPC-{int(args.mpc_horizon)}"

    print("Diagnostic cases:", [case.case_id for case in cases])
    print("Scenarios:", scenario_ids)
    print("Methods:", methods)
    print("forecast_case:", args.forecast_case)
    print("actual_case:", args.actual_case)
    print("mpc_horizon:", args.mpc_horizon)
    print("mpc_output_method:", mpc_method_name)
    print("relaxed_ramp_value:", args.relaxed_ramp_value)
    print("reuse_original:", args.reuse_original)
    print("original_mpc3_input:", args.original_mpc3_input)
    print("original_perfect_input:", args.original_perfect_input)
    print("reuse_perfect_global:", args.reuse_perfect_global)
    print("perfect_cache_input:", args.perfect_cache_input)
    print("mpc_time_limit_s:", args.mpc_time_limit_s)
    print("perfect_time_limit_s:", args.perfect_time_limit_s)
    print("mip_gap:", args.mip_gap)
    print("max_steps:", args.max_steps)
    print("include_tail:", not args.no_tail)

    if args.dry_run:
        return

    rows: List[Dict[str, Any]] = []
    perfect_cache = (
        load_perfect_cache_index(args.perfect_cache_input)
        if args.reuse_perfect_global and "Perfect-Global" in methods
        else {}
    )
    for case in cases:
        case_rows: List[Dict[str, Any]] = []

        if case.case_id == "original" and args.reuse_original:
            case_rows = load_original_cached_rows(
                args=args,
                scenario_ids=scenario_ids,
                methods=methods,
                mpc_method_name=mpc_method_name,
            )
            rows.extend(case_rows)
            save_case_outputs(case=case, case_rows=case_rows, args=args)
            print("[Reuse] original case loaded from existing formal results.")
            continue

        customizer = make_solver_customizer(case, args.relaxed_ramp_value)
        for scenario_id in scenario_ids:
            scenario_rows: List[Dict[str, Any]] = []
            if "MPC-3" in methods:
                row = run_one_scenario_33(
                    method=mpc_method_name,
                    solver_kind="pure_mpc_clean",
                    scenario_id=scenario_id,
                    forecast_case=args.forecast_case,
                    actual_case=args.actual_case,
                    horizon=int(args.mpc_horizon),
                    gamma_Q=None,
                    split=None,
                    max_steps=args.max_steps,
                    include_tail=not args.no_tail,
                    quiet_solver=not args.verbose_solver,
                    time_limit_s=args.mpc_time_limit_s,
                    mip_gap=args.mip_gap,
                    solver_customizer=customizer,
                )
                scenario_rows.append(
                    decorate_row(row, case=case, relaxed_ramp_value=args.relaxed_ramp_value)
                )

            if "Perfect-Global" in methods:
                if args.reuse_perfect_global:
                    key = (case.case_id, int(scenario_id), args.actual_case)
                    if key not in perfect_cache:
                        raise RuntimeError(
                            "Missing cached Perfect-Global result for "
                            f"case={case.case_id}, scenario={scenario_id}, "
                            f"actual_case={args.actual_case} in "
                            f"{resolve_result_path(args.perfect_cache_input)}"
                        )
                    row = dict(perfect_cache[key])
                    row["source_forecast_case"] = row.get("forecast_case", "")
                    row["forecast_case"] = args.forecast_case
                    row["source_cache"] = args.perfect_cache_input
                else:
                    row = run_one_perfect_global_33(
                        scenario_id=scenario_id,
                        actual_case=args.actual_case,
                        quiet_solver=not args.verbose_solver,
                        time_limit_s=args.perfect_time_limit_s,
                        mip_gap=args.mip_gap,
                        solver_customizer=customizer,
                    )
                scenario_rows.append(
                    decorate_row(row, case=case, relaxed_ramp_value=args.relaxed_ramp_value)
                )

            add_pair_gap(scenario_rows)
            for row in scenario_rows:
                rows.append(row)
                case_rows.append(row)
                print(
                    f"{row['diagnostic_case']}, {row['method']}, scenario={scenario_id}, "
                    f"completed={row.get('completed')}, cost={row.get('cost')}, "
                    f"perfect={row.get('perfect_cost')}, gap={row.get('gap_percent')}, "
                    f"time={row.get('solve_time_s')}, "
                    f"status={row.get('gurobi_status')}, "
                    f"fail={row.get('solver_fail_count')}"
                )

        save_case_outputs(case=case, case_rows=case_rows, args=args)

    add_pair_gap(rows)
    raw_path = resolve_result_path(args.output)
    write_rows(raw_path, rows)
    summary_rows = build_summary_rows(rows, cases)
    summary_path = resolve_result_path(args.summary_output)
    write_rows(summary_path, summary_rows)
    write_paper_table(summary_rows, resolve_result_path(args.table_output_dir))
    print("Raw results:", raw_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
