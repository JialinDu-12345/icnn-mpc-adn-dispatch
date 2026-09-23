from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Optional

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import SPLITS, validate_actual_case              
from experiments_v2.policy_baselines.policy_adapter_33 import (              
    PolicyAdapter33,
    available_policy_methods,
)
from experiments_v2.result_logger import ResultLogger              


def parse_csv_list(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_scenarios(raw: str, scenario_count: Optional[int]) -> List[int]:
    if raw.strip().lower() == "all":
        scenario_ids = list(SPLITS["33"]["test_id"])
    else:
        scenario_ids = [int(item) for item in parse_csv_list(raw)]
    if scenario_count is not None:
        scenario_ids = scenario_ids[:scenario_count]
    return scenario_ids


def parse_methods(raw: str) -> List[str]:
    available = available_policy_methods()
    if raw.strip().lower() == "all":
        return available

    methods: List[str] = []
    seen = set()
    aliases = {method.lower(): method for method in available}
    aliases.update({"sac-icnn": "SAC-ICNN", "sacicnn": "SAC-ICNN", "saclag": "SACLag"})
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
    parser = argparse.ArgumentParser(description="Run 33-bus RL policy baselines.")
    parser.add_argument("--scenarios", default="all")
    parser.add_argument("--scenario-count", type=int, default=None)
    parser.add_argument("--methods", default="SAC,SAC-ICNN,SACLag")
    parser.add_argument("--actual-case", default="id_actual")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument(
        "--eval-seed",
        type=int,
        default=None,
        help=(
            "Random seed used before each policy rollout. "
            "Use --stochastic --eval-seed 0 for legacy-style sampled-action evaluation."
        ),
    )
    parser.add_argument("--verbose-legacy-init", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default="formal/step7_33_policy_baselines_raw.csv")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    validate_actual_case(args.actual_case)

    scenario_ids = parse_scenarios(args.scenarios, args.scenario_count)
    methods = parse_methods(args.methods)

    print("Scenarios:", scenario_ids)
    print("Methods:", methods)
    print("actual_case:", args.actual_case)
    print("max_steps:", args.max_steps)
    print("device:", args.device)
    print("stochastic:", args.stochastic)
    print("eval_seed:", args.eval_seed)

    if args.dry_run:
        return

    adapters = {
        method: PolicyAdapter33(
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
                eval_seed=args.eval_seed,
            )
            logger.add_row(row)
            print(
                f"{method}, scenario={scenario_id}, "
                f"completed={row.get('completed')}, "
                f"steps={row.get('executed_steps')}, "
                f"cost={float(row.get('cost') or 0.0):.6f}, "
                f"vio={float(row.get('voltage_violation') or 0.0):.6f}, "
                f"time={float(row.get('solve_time_s') or 0.0):.4f}, "
                f"fail={row.get('solver_fail_count')}, "
                f"reason={row.get('incomplete_reason')}"
            )

    path = logger.save()
    print("Saved:", path)


if __name__ == "__main__":
    main()
