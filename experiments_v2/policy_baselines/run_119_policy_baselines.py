from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import validate_actual_case              
from experiments_v2.policy_baselines.policy_adapter_119 import (              
    PolicyAdapter119,
    available_policy_methods,
)
from experiments_v2.result_logger import ResultLogger              
from experiments_v2.utils.scenario_selection import resolve_scenarios              


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_methods(raw: str) -> List[str]:
    available = available_policy_methods()
    if raw.strip().lower() == "all":
        return available

    methods: List[str] = []
    seen = set()
    aliases = {method.lower(): method for method in available}
    aliases.update({"sac-icnn": "SAC-ICNN", "sacicnn": "SAC-ICNN"})
    for item in parse_csv_list(raw):
        method = aliases.get(item.lower(), item)
        if method not in available:
            raise ValueError(f"Unknown policy '{item}'. Available: {available}")
        if method in seen:
            continue
        methods.append(method)
        seen.add(method)
    return methods


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run 119-bus RL policy baselines.")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument(
        "--scenario-file",
        default=None,
        help="JSON/CSV file under results_v2 containing selected scenarios. Overrides --scenarios.",
    )
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--methods", default="SAC,SAC-ICNN")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--verbose-legacy-init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step10_119_policy_baselines_raw.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_actual_case(args.actual_case)

    scenario_ids = resolve_scenarios(
        system="119",
        scenarios=args.scenarios,
        scenario_file=args.scenario_file,
        scenario_count=args.scenario_count,
    )
    methods = parse_methods(args.methods)

    print("Scenarios:", scenario_ids)
    print("Methods:", methods)
    print("actual_case:", args.actual_case)
    print("scenario_file:", args.scenario_file or "")
    print("max_steps:", args.max_steps)
    print("device:", args.device)
    print("stochastic:", args.stochastic)

    if args.dry_run:
        return

    adapters = {
        method: PolicyAdapter119(
            method=method,
            device=args.device,
            quiet_legacy_init=not args.verbose_legacy_init,
        )
        for method in methods
    }

    logger = ResultLogger(args.output)
    for scenario_id in scenario_ids:
        for method in methods:
            row = adapters[method].run_one_day(
                scenario_id=scenario_id,
                actual_case=args.actual_case,
                max_steps=args.max_steps,
                stochastic=args.stochastic,
            )
            row["scenario_selection_file"] = args.scenario_file or ""
            row["scenario_selection_rule"] = (
                "perfect_global_feasible_certified_subset" if args.scenario_file else ""
            )
            logger.add_row(row)
            print(
                f"{method}, scenario={scenario_id}, "
                f"completed={row.get('completed')}, "
                f"steps={row.get('executed_steps')}, "
                f"cost={float(row.get('cost') or 0.0):.6f}, "
                f"time={float(row.get('solve_time_s') or 0.0):.4f}, "
                f"fail={row.get('solver_fail_count')}, "
                f"reason={row.get('incomplete_reason')}"
            )

    path = logger.save()
    print("Saved:", path)


if __name__ == "__main__":
    main()
