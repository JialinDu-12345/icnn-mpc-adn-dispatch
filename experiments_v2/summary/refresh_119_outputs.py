from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments_v2.config import RESULT_DIR              


SELECTED_JSON = "formal/selected_119_inline_operable20.json"
SCREENING_RAW = "formal/step10_119_inline_screen_log.csv"
SCREENING_SUMMARY = "formal/step10_119_inline_screening_summary.csv"
SELECTED_MAIN_RAW = "formal/step10_119_main_inline20_raw.csv"
SELECTED_PERFECT_RAW = SELECTED_MAIN_RAW
SELECTED_SUMMARY = "formal/step10_119_scalability_inline20_summary.csv"
SELECTED_SUMMARY_WITH_GAP = "formal/step10_119_scalability_inline20_summary_with_gap.csv"
SELECTED_MAIN_TABLE = "formal/step10_119_main_table_inline20.csv"


def result_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return RESULT_DIR / path


def exists(raw: str) -> bool:
    return result_path(raw).exists()


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
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    if result.returncode != 0:
        message = f"Command failed with code {result.returncode}: {' '.join(cmd)}"
        if required:
            raise RuntimeError(message)
        print("[WARN]", message)
        return False
    return True


def refresh_screening(strict: bool) -> bool:
    if not exists(SCREENING_RAW):
        print("[SKIP] Inline screening log not found:", SCREENING_RAW)
        return not strict

    return run_module(
        "experiments_v2.summary.summarize_119_perfect_screening",
        ["--input", SCREENING_RAW, "--output", SCREENING_SUMMARY],
        required=strict,
    )


def validate_selected20(strict: bool, target_count: int, mip_gap: float) -> bool:
    missing = [
        raw
        for raw in [SELECTED_JSON, SELECTED_PERFECT_RAW]
        if not exists(raw)
    ]
    if missing:
        print("[SKIP] selected20 files are not ready yet:", missing)
        return not strict

    if not strict:
        try:
            payload = json.loads(result_path(SELECTED_JSON).read_text(encoding="utf-8-sig"))
            selected = payload.get("selected_scenarios", payload.get("scenario_ids", []))
            if len(selected) != target_count:
                print(
                    "[SKIP] selected20 validation is not ready yet: "
                    f"selected_count={len(selected)}/{target_count}"
                )
                return True
        except Exception as exc:
            print("[SKIP] selected20 validation is not ready yet:", exc)
            return True

    return run_module(
        "experiments_v2.summary.validate_119_selected20",
        [
            "--selected-json",
            SELECTED_JSON,
            "--selected-perfect-raw",
            SELECTED_PERFECT_RAW,
            "--target-count",
            str(target_count),
            "--mip-gap",
            str(mip_gap),
        ],
        required=strict,
    )


def refresh_scalability(strict: bool) -> bool:
    missing = [
        raw
        for raw in [SELECTED_MAIN_RAW, SELECTED_PERFECT_RAW]
        if not exists(raw)
    ]
    if missing:
        print("[SKIP] inline20 main/perfect raw not ready:", missing)
        return not strict

    inputs = (
        SELECTED_MAIN_RAW
        if SELECTED_MAIN_RAW == SELECTED_PERFECT_RAW
        else f"{SELECTED_MAIN_RAW},{SELECTED_PERFECT_RAW}"
    )
    return run_module(
        "experiments_v2.summary.summarize_119_scalability",
        [
            "--inputs",
            inputs,
            "--output",
            SELECTED_SUMMARY,
            "--strict",
        ],
        required=strict,
    )


def refresh_gap(strict: bool) -> bool:
    missing = [
        raw
        for raw in [SELECTED_SUMMARY, SELECTED_MAIN_RAW, SELECTED_PERFECT_RAW]
        if not exists(raw)
    ]
    if missing:
        print("[SKIP] gap inputs not ready:", missing)
        return not strict

    return run_module(
        "experiments_v2.summary.add_gap_to_perfect_119",
        [
            "--summary",
            SELECTED_SUMMARY,
            "--main-raw",
            SELECTED_MAIN_RAW,
            "--perfect-raw",
            SELECTED_PERFECT_RAW,
            "--output",
            SELECTED_SUMMARY_WITH_GAP,
        ],
        required=strict,
    )


def refresh_main_table(strict: bool) -> bool:
    if not exists(SELECTED_SUMMARY_WITH_GAP):
        print("[SKIP] summary_with_gap not ready:", SELECTED_SUMMARY_WITH_GAP)
        return not strict

    return run_module(
        "experiments_v2.summary.merge_119_main_table",
        [
            "--inputs",
            SELECTED_SUMMARY_WITH_GAP,
            "--output",
            SELECTED_MAIN_TABLE,
            "--strict",
        ],
        required=strict,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh 119-bus selected20 outputs.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if selected20 inputs are missing or any step fails.",
    )
    parser.add_argument("--skip-screening-summary", action="store_true")
    parser.add_argument("--skip-validate", action="store_true")
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    print("Refreshing 119 selected20 outputs")
    print("RESULT_DIR:", RESULT_DIR)

    if not args.skip_screening_summary:
        refresh_screening(strict=False)

    if not args.skip_validate:
        validate_selected20(
            strict=args.strict,
            target_count=args.target_count,
            mip_gap=args.mip_gap,
        )

    refresh_scalability(strict=args.strict)
    refresh_gap(strict=args.strict)
    refresh_main_table(strict=args.strict)

    print("\nDone.")


if __name__ == "__main__":
    main()
