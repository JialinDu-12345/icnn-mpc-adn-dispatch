from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_v2.config import RESULT_DIR              


def exists(relative_path: str) -> bool:
    return (RESULT_DIR / relative_path).exists()


def run_module(module: str, args: list[str] | None = None, required: bool = True) -> bool:
    cmd = [sys.executable, "-m", module, *(args or [])]
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(PROJECT_ROOT)
        if not old_pythonpath
        else str(PROJECT_ROOT) + os.pathsep + old_pythonpath
    )

    print("\n[RUN]", " ".join(cmd), flush=True)
    print("[CWD]", PROJECT_ROOT, flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    if result.returncode != 0:
        message = f"Command failed with code {result.returncode}: {' '.join(cmd)}"
        if required:
            raise RuntimeError(message)
        print("[WARN]", message)
        return False
    return True


def resolve_main_table_input() -> str | None:
    candidates = [
        "formal/step8_33_main_table_with_perfect_gap.csv",
        "formal/step8_33_main_table_with_perfect.csv",
        "formal/step8_33_main_table_refreshed.csv",
        "formal/step8_33_main_table.csv",
    ]
    for candidate in candidates:
        if exists(candidate):
            return candidate
    return None


def refresh_perfect() -> str:
    if not exists("formal/step7b_33_perfect_global_raw.csv"):
        print("[SKIP] Perfect-Global raw not found yet.")
        return ""

    run_module(
        "experiments_v2.summary.summarize_perfect_global",
        [
            "--input",
            "formal/step7b_33_perfect_global_raw.csv",
            "--output",
            "formal/step7b_33_perfect_global_summary.csv",
        ],
    )
    return "formal/step7b_33_perfect_global_summary.csv"


def refresh_forecast_sensitivity() -> None:
    if not exists("formal/step5_33_forecast_sensitivity_raw.csv"):
        print("[SKIP] Forecast sensitivity raw not found.")
        return

    run_module(
        "experiments_v2.summary.summarize_main_model_based",
        [
            "--input",
            "formal/step5_33_forecast_sensitivity_raw.csv",
            "--output",
            "formal/step5_33_forecast_sensitivity_summary.csv",
        ],
    )
    for baseline in ["MPC-3", "MPC-7"]:
        output = (
            "formal/step5_33_pairwise_mpc3_vs_proposed_by_case.csv"
            if baseline == "MPC-3"
            else "formal/step5_33_pairwise_mpc7_vs_proposed_by_case.csv"
        )
        run_module(
            "experiments_v2.summary.summarize_pairwise_by_case",
            [
                "--input",
                "formal/step5_33_forecast_sensitivity_raw.csv",
                "--baseline",
                baseline,
                "--target",
                "Proposed-H3",
                "--output",
                output,
            ],
        )


def refresh_nn_embedding() -> None:
    jobs = [
        (
            "formal/step6_33_nn_embedding_raw.csv",
            "formal/step6_33_nn_embedding_summary.csv",
        ),
        (
            "formal/step6_33_stdnn_only_raw.csv",
            "formal/step6_33_stdnn_only_summary.csv",
        ),
    ]
    ran_any = False
    for input_path, output_path in jobs:
        if not exists(input_path):
            print("[MISS]", input_path)
            continue
        ran_any = True
        run_module(
            "experiments_v2.summary.summarize_nn_embedding",
            [
                "--input",
                input_path,
                "--output",
                output_path,
            ],
        )
    if not ran_any:
        print("[SKIP] NN embedding raw not found.")


def merge_main_table(perfect_summary: str, strict: bool) -> str:
    inputs = []
    if perfect_summary:
        inputs.append(perfect_summary)

    for relative_path in [
        "formal/step7_33_oracle_mpc_summary.csv",
        "formal/step4_33_main_model_based_summary.csv",
        "formal/step6_33_stdnn_only_summary.csv",
        "formal/step6_33_nn_embedding_summary.csv",
        "formal/step7_33_policy_baselines_summary.csv",
        "formal/step7_33_saclag_stochastic_seed0_test_summary.csv",
    ]:
        if exists(relative_path):
            inputs.append(relative_path)
        else:
            print("[MISS]", relative_path)

    if not inputs:
        print("[SKIP] No main-table summary inputs found.")
        return ""

    output = (
        "formal/step8_33_main_table_with_perfect.csv"
        if perfect_summary
        else "formal/step8_33_main_table_refreshed.csv"
    )
    merge_args = ["--inputs", ",".join(inputs), "--output", output]
    if strict:
        merge_args.append("--strict")
    run_module("experiments_v2.summary.merge_33_main_table", merge_args)
    return output


def add_perfect_gap(main_table: str, strict: bool) -> str:
    if not main_table or not exists(main_table):
        return ""
    if not exists("formal/step7b_33_perfect_global_raw.csv"):
        print("[SKIP] Perfect-Global raw not found; gap_to_perfect not calculated.")
        return ""

    output = "formal/step8_33_main_table_with_perfect_gap.csv"
    args = [
        "--input",
        main_table,
        "--perfect-raw",
        "formal/step7b_33_perfect_global_raw.csv",
        "--output",
        output,
    ]
    run_module("experiments_v2.summary.add_gap_to_perfect_33", args)
    return output


def refresh_paper_tables(table_input: str) -> None:
    if not table_input or not exists(table_input):
        print("[SKIP] Main table input not found; paper tables not generated.")
        return

    run_module(
        "experiments_v2.summary.make_33_paper_tables",
        [
            "--main-table",
            table_input,
            "--forecast-pairwise",
            "formal/step5_33_pairwise_mpc3_vs_proposed_by_case.csv",
            "--nn-summary",
            "formal/step6_33_nn_embedding_summary.csv",
            "--output-dir",
            "paper_tables/33",
        ],
        required=False,
    )


def refresh_plots(table_input: str) -> None:
    if table_input and exists(table_input):
        run_module(
            "experiments_v2.plots.plot_33_main_results",
            ["--input", table_input, "--output-dir", "paper_figures/33"],
            required=False,
        )
    else:
        print("[SKIP] Main table input not found; main plots not generated.")

    if exists("formal/step5_33_forecast_sensitivity_summary.csv"):
        run_module(
            "experiments_v2.plots.plot_33_forecast_sensitivity",
            [
                "--summary",
                "formal/step5_33_forecast_sensitivity_summary.csv",
                "--pairwise",
                "formal/step5_33_pairwise_mpc3_vs_proposed_by_case.csv",
                "--output-dir",
                "paper_figures/33",
            ],
            required=False,
        )
    else:
        print("[SKIP] Forecast sensitivity summary not found; forecast plots not generated.")

    if exists("formal/step6_33_nn_embedding_summary.csv"):
        run_module(
            "experiments_v2.plots.plot_33_nn_embedding",
            [
                "--input",
                "formal/step6_33_nn_embedding_summary.csv",
                "--output-dir",
                "paper_figures/33",
            ],
            required=False,
        )
    else:
        print("[SKIP] NN embedding summary not found; NN plots not generated.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh all 33-bus summaries, tables, and figures.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    perfect_summary = refresh_perfect()
    refresh_forecast_sensitivity()
    refresh_nn_embedding()
    main_table = merge_main_table(perfect_summary, strict=args.strict)
    gap_table = add_perfect_gap(main_table, strict=args.strict)

    table_input = gap_table or main_table or resolve_main_table_input()
    if table_input:
        refresh_paper_tables(table_input)

    if not args.skip_plots:
        refresh_plots(table_input or "")

if __name__ == "__main__":
    main()
