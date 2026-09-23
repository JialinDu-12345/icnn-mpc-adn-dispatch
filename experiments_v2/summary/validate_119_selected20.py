from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return RESULT_DIR / path


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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


def read_selected_scenarios(path: Path) -> list[int]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if "selected_scenarios" in payload:
        selected = payload["selected_scenarios"]
    elif "scenario_ids" in payload:
        selected = payload["scenario_ids"]
    else:
        raise KeyError(f"{path} must contain 'selected_scenarios' or 'scenario_ids'.")
    return [int(value) for value in selected]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate the 119-bus selected20 subset.")
    parser.add_argument(
        "--selected-json",
        default="formal/selected_119_inline_operable20.json",
    )
    parser.add_argument(
        "--selected-perfect-raw",
        default="formal/step10_119_main_inline20_raw.csv",
    )
    parser.add_argument("--target-count", type=int, default=20)
    parser.add_argument("--mip-gap", type=float, default=0.001)
    parser.add_argument(
        "--allow-extra-perfect-rows",
        action="store_true",
        help="Warn instead of failing if selected Perfect raw contains extra scenarios.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    json_path = resolve_path(args.selected_json)
    perfect_path = resolve_path(args.selected_perfect_raw)

    if not json_path.exists():
        raise FileNotFoundError(json_path)
    if not perfect_path.exists():
        raise FileNotFoundError(perfect_path)

    selected = read_selected_scenarios(json_path)
    if len(selected) != args.target_count:
        raise RuntimeError(
            f"selected_scenarios has {len(selected)} scenarios, "
            f"but target-count is {args.target_count}."
        )
    if len(set(selected)) != len(selected):
        raise RuntimeError("selected_scenarios contains duplicate scenario ids.")

    df = pd.read_csv(perfect_path, dtype={"gamma_Q": str})
    if "method" not in df.columns:
        raise KeyError(f"{perfect_path} must contain a 'method' column.")
    if "scenario_id" not in df.columns:
        raise KeyError(f"{perfect_path} must contain a 'scenario_id' column.")

    df = df[df["method"].astype(str) == "Perfect-Global"].copy()
    df["scenario_id_int"] = pd.to_numeric(df["scenario_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["scenario_id_int"]).copy()
    df["scenario_id_int"] = df["scenario_id_int"].astype(int)

    expected = set(selected)
    got = set(df["scenario_id_int"].tolist())
    missing = sorted(expected - got)
    extra = sorted(got - expected)
    if missing:
        raise RuntimeError(f"Selected Perfect raw misses scenarios: {missing}")
    if extra:
        message = f"Selected Perfect raw has extra scenarios: {extra}"
        if args.allow_extra_perfect_rows:
            print("[WARN]", message)
        else:
            raise RuntimeError(message)

    bad_rows: list[tuple[int, str]] = []
    selected_df = df[df["scenario_id_int"].isin(expected)].copy()
    if selected_df["scenario_id_int"].duplicated().any():
        duplicated = sorted(
            selected_df.loc[selected_df["scenario_id_int"].duplicated(), "scenario_id_int"]
            .astype(int)
            .tolist()
        )
        raise RuntimeError(f"Selected Perfect raw has duplicate scenarios: {duplicated}")

    for _, row in selected_df.iterrows():
        sid = int(row["scenario_id_int"])
        completed = parse_bool(row.get("completed"))
        certified = parse_bool(row.get("certified_global"))
        fail = int(parse_float(row.get("solver_fail_count"), default=0.0))
        bin_vars = parse_float(row.get("mean_num_bin_vars"))
        mip_gap = parse_float(row.get("mean_mip_gap"))

        reason = []
        if not completed:
            reason.append("not completed")
        if not certified:
            reason.append("not certified_global")
        if fail != 0:
            reason.append(f"solver_fail_count={fail}")
        if math.isfinite(bin_vars) and abs(bin_vars) > 1e-9:
            reason.append(f"mean_num_bin_vars={bin_vars}")
        if math.isfinite(mip_gap) and mip_gap > args.mip_gap + 1e-9:
            reason.append(f"mean_mip_gap={mip_gap}")

        if reason:
            bad_rows.append((sid, "; ".join(reason)))

    if bad_rows:
        for sid, reason in bad_rows:
            print(f"[BAD] scenario={sid}: {reason}")
        raise RuntimeError("Some selected Perfect-Global rows are invalid.")

    print("Selected 119 scenarios are valid.")
    print("selected_scenarios:", selected)
    print("selected_count:", len(selected))
    print("selected_json:", json_path)
    print("selected_perfect_raw:", perfect_path)


if __name__ == "__main__":
    main()
