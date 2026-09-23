from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import DATA_DIRS, RESULT_DIR, SPLITS, validate_actual_case              
from experiments_v2.policy_baselines.policy_adapter_33 import PolicyAdapter33              


def resolve_result_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return RESULT_DIR / path


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_scenarios(raw: str) -> List[int]:
    text = raw.strip().lower()
    if text == "all":
        return list(SPLITS["33"]["historical"] + SPLITS["33"]["val"] + SPLITS["33"]["test_id"])
    if text in SPLITS["33"]:
        return list(SPLITS["33"][text])
    if ":" in text:
        start_text, end_text = text.split(":", 1)
        start = int(start_text)
        end = int(end_text)
        if end < start:
            raise ValueError(f"Invalid scenario range: {raw}")
        return list(range(start, end + 1))
    return [int(item) for item in parse_csv_list(text)]


def filter_existing_scenarios(
    scenario_ids: List[int],
    *,
    strict: bool,
) -> List[int]:
    existing: List[int] = []
    missing: List[int] = []
    for scenario_id in scenario_ids:
        path = DATA_DIRS["33"] / f"results_{int(scenario_id)}.npz"
        if path.exists():
            existing.append(int(scenario_id))
        else:
            missing.append(int(scenario_id))

    if missing and strict:
        preview = ", ".join(str(item) for item in missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} candidate scenario files under {DATA_DIRS['33']}. "
            f"First missing IDs: {preview}"
        )
    if missing:
        preview = ", ".join(str(item) for item in missing[:10])
        suffix = " ..." if len(missing) > 10 else ""
        print(
            f"[Skipped] {len(missing)} candidate scenarios have no copied data file. "
            f"First missing IDs: {preview}{suffix}"
        )
    if not existing:
        raise RuntimeError("No existing 33-bus scenario files were found for the requested scan.")
    return existing


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_float(value: object, default: float = math.nan) -> float:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    try:
        return float(text)
    except (TypeError, ValueError):
        return default


def parse_step_violations(row: Dict[str, object]) -> List[float]:
    raw = row.get("step_cost_constraints_json", "")
    if raw:
        try:
            values = json.loads(str(raw))
            return [float(value) for value in values if math.isfinite(float(value))]
        except (TypeError, ValueError, json.JSONDecodeError):
            pass

    mean_vio = safe_float(row.get("mean_cost_constraint"))
    executed_steps = int(safe_float(row.get("executed_steps"), 0.0))
    if math.isfinite(mean_vio) and executed_steps > 0:
        return [mean_vio] * executed_steps
    return []


def compute_violation_metrics(row: Dict[str, object], threshold: float) -> Dict[str, object]:
    step_violations = parse_step_violations(row)
    positive_steps = [max(0.0, value) for value in step_violations]

    if positive_steps:
        cumulative_vio = float(sum(positive_steps))
        max_step_vio = float(max(positive_steps))
        violated_steps = int(sum(1 for value in positive_steps if value > threshold))
    else:
        cumulative_vio = safe_float(row.get("voltage_violation"))
        max_step_vio = math.nan
        violated_steps = int(cumulative_vio > threshold) if math.isfinite(cumulative_vio) else 0

    return {
        "saclag_cumulative_vio": cumulative_vio,
        "saclag_max_step_vio": max_step_vio,
        "saclag_violated_steps": violated_steps,
    }


def csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("No SACLag scan rows to write.")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def save_selected_json(
    *,
    path: Path,
    selected_rows: List[Dict[str, object]],
    threshold: float,
    candidate_scenarios: List[int],
    stochastic: bool,
    eval_seed: Optional[int],
) -> None:
    selected = [int(row["scenario_id"]) for row in selected_rows]
    payload = {
        "system": "33",
        "selected_scenarios": selected,
        "selection_rule": (
            "Diagnostic stress cases selected because the copied pretrained SACLag "
            "policy exhibits positive voltage-limit violations. These cases do not "
            "replace the fixed held-out scenarios used in the main 33-bus table."
            if selected
            else "No copied 33-bus candidate scenario exceeded the positive voltage-violation threshold."
        ),
        "method_used_for_screening": "SACLag",
        "policy_eval_mode": "stochastic_sampled_action"
        if stochastic
        else "deterministic_mean_action",
        "policy_eval_seed": None if eval_seed is None else int(eval_seed),
        "violation_metric": (
            "sum(max(step voltage violation, 0)), recomputed from "
            "env.net.res_bus.vm_pu with vmin=0.94 and vmax=1.06"
        ),
        "violation_threshold": float(threshold),
        "candidate_scenarios": [int(scenario_id) for scenario_id in candidate_scenarios],
        "note": (
            "Use this JSON only for the additional SACLag stress diagnostic table, "
            "not for the average-performance main table."
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_stress_tex(path: Path, selected_rows: List[Dict[str, object]]) -> None:
    protocol_text = ""
    if selected_rows:
        eval_mode = str(selected_rows[0].get("policy_eval_mode", "")).strip()
        eval_seed = str(selected_rows[0].get("policy_eval_seed", "")).strip()
        if eval_mode:
            protocol_text = f" The screening protocol is {eval_mode}"
            if eval_seed:
                protocol_text += f" with eval seed {eval_seed}"
            protocol_text += "."

    lines = [
        r"\begin{table}[!t]",
        r"\caption{SACLag Voltage-Violation Diagnostic Cases on the 33-Bus System}",
        r"\label{tab:saclag_violation_stress_33}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.2pt}",
        r"\begin{tabular}{cccc}",
        r"\toprule",
        r"Scenario & Cost (\$) & Vio. (p.u.) & Violated steps \\",
        r"\midrule",
    ]
    if selected_rows:
        for row in selected_rows:
            scenario_id = int(row["scenario_id"])
            cost = safe_float(row.get("cost"))
            cumulative_vio = safe_float(row.get("saclag_cumulative_vio"))
            violated_steps = int(safe_float(row.get("saclag_violated_steps"), 0.0))
            lines.append(f"{scenario_id} & {cost:.2f} & {cumulative_vio:.4f} & {violated_steps} \\\\")
    else:
        lines.append(r"\multicolumn{4}{c}{No positive SACLag voltage-violation cases found.} \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{0.5mm}",
            r"\parbox{0.98\linewidth}{\footnotesize",
            (
                r"These diagnostic cases are selected from a larger candidate scenario pool "
                r"because the pretrained SACLag policy exhibits positive voltage-limit "
                r"violations. They are not used to replace the fixed held-out test scenarios "
                r"in the main 33-bus table."
                + protocol_text
                if selected_rows
                else r"No positive SACLag voltage-violation cases were found in the copied "
                r"candidate scenario pool. This diagnostic does not modify the fixed held-out "
                r"test scenarios in the main 33-bus table."
            ),
            r"}",
            r"\end{table}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search 33-bus diagnostic scenarios where SACLag has voltage violations."
    )
    parser.add_argument(
        "--scenarios",
        default="0:499",
        help="Candidate scenario IDs: '0:499', '80,81,82', 'historical', 'test_id', or 'all'.",
    )
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=1e-6)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Evaluate SACLag using sampled stochastic actions, consistent with the legacy test script.",
    )
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Random seed used before each SACLag rollout. "
            "Use --stochastic --eval-seed 0 for legacy-style reproducible evaluation."
        ),
    )
    parser.add_argument("--verbose-legacy-init", action="store_true")
    parser.add_argument(
        "--strict-scenario-files",
        action="store_true",
        help="Fail instead of skipping candidate IDs whose copied scenario files are missing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step9_33_saclag_violation_scan.csv")
    parser.add_argument(
        "--selected-json",
        default="formal/selected_33_saclag_violation_stress5.json",
    )
    parser.add_argument(
        "--tex-output",
        default="paper_tables/33/table_33_saclag_violation_stress.tex",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    validate_actual_case(args.actual_case)

    requested_scenario_ids = parse_scenarios(args.scenarios)
    scenario_ids = filter_existing_scenarios(
        requested_scenario_ids,
        strict=args.strict_scenario_files,
    )
    print("Scanning SACLag violation cases")
    print("Scenarios:", scenario_ids)
    print("requested_scenario_count:", len(requested_scenario_ids))
    print("scanned_scenario_count:", len(scenario_ids))
    print("actual_case:", args.actual_case)
    print("top_k:", args.top_k)
    print("threshold:", args.threshold)
    print("max_steps:", args.max_steps)
    print("stochastic:", args.stochastic)
    print("eval_seed:", args.eval_seed)
    print("output:", resolve_result_path(args.output))

    if args.dry_run:
        return

    adapter = PolicyAdapter33(
        method="SACLag",
        device=args.device,
        quiet_legacy_init=not args.verbose_legacy_init,
    )

    rows: List[Dict[str, object]] = []
    for scenario_id in scenario_ids:
        try:
            row = adapter.run_one_day(
                scenario_id=scenario_id,
                actual_case=args.actual_case,
                max_steps=args.max_steps,
                stochastic=args.stochastic,
                eval_seed=args.eval_seed,
            )
            out = dict(row)
            out.update(compute_violation_metrics(out, threshold=args.threshold))
            out["screening_method"] = "SACLag"
            out["diagnostic_stress_scan"] = True
            rows.append(out)
            print(
                f"SACLag scenario={scenario_id}, completed={out.get('completed')}, "
                f"cost={safe_float(out.get('cost'), 0.0):.6f}, "
                f"vio={safe_float(out.get('saclag_cumulative_vio'), 0.0):.6f}, "
                f"violated_steps={out.get('saclag_violated_steps')}"
            )
        except Exception as exc:
            out = {
                "system": "33",
                "method": "SACLag",
                "scenario_id": int(scenario_id),
                "actual_case": args.actual_case,
                "completed": False,
                "incomplete_reason": "policy_failure",
                "failure_message": repr(exc),
                "saclag_cumulative_vio": math.nan,
                "saclag_max_step_vio": math.nan,
                "saclag_violated_steps": 0,
                "screening_method": "SACLag",
                "diagnostic_stress_scan": True,
            }
            rows.append(out)
            print(f"[FAILED] SACLag scenario={scenario_id}: {exc}")

    output_path = resolve_result_path(args.output)
    write_csv(output_path, rows)
    print("Saved scan raw:", output_path)

    completed_vio_rows = [
        row
        for row in rows
        if parse_bool(row.get("completed"))
        and safe_float(row.get("saclag_cumulative_vio"), 0.0) > args.threshold
    ]
    completed_vio_rows.sort(
        key=lambda row: safe_float(row.get("saclag_cumulative_vio"), -math.inf),
        reverse=True,
    )

    if not completed_vio_rows:
        print("No completed SACLag violation cases found above threshold.")
        print("Try increasing --scenarios, for example --scenarios 0:999.")
        selected_json_path = resolve_result_path(args.selected_json)
        tex_output_path = resolve_result_path(args.tex_output)
        save_selected_json(
            path=selected_json_path,
            selected_rows=[],
            threshold=args.threshold,
            candidate_scenarios=scenario_ids,
            stochastic=args.stochastic,
            eval_seed=args.eval_seed,
        )
        write_stress_tex(tex_output_path, [])
        print("Saved empty selected stress-case JSON:", selected_json_path)
        print("Saved no-violation LaTeX table:", tex_output_path)
        return

    selected_rows = completed_vio_rows[: max(0, int(args.top_k))]
    print("Top SACLag violation cases:")
    for row in selected_rows:
        print(
            {
                "scenario_id": row.get("scenario_id"),
                "cost": row.get("cost"),
                "saclag_cumulative_vio": row.get("saclag_cumulative_vio"),
                "saclag_max_step_vio": row.get("saclag_max_step_vio"),
                "saclag_violated_steps": row.get("saclag_violated_steps"),
            }
        )

    selected_json_path = resolve_result_path(args.selected_json)
    tex_output_path = resolve_result_path(args.tex_output)
    save_selected_json(
        path=selected_json_path,
        selected_rows=selected_rows,
        threshold=args.threshold,
        candidate_scenarios=scenario_ids,
        stochastic=args.stochastic,
        eval_seed=args.eval_seed,
    )
    write_stress_tex(tex_output_path, selected_rows)
    print("Saved selected stress-case JSON:", selected_json_path)
    print("Saved LaTeX table:", tex_output_path)


if __name__ == "__main__":
    main()
