from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable, List, Optional

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


METHOD_ORDER = ["MPC-3", "Proposed-H3"]


def to_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        text = str(value).strip()
        if not text:
            return default
        return float(text)
    except Exception:
        return default


def fmt_int(value: Any) -> str:
    numeric = to_float(value)
    if not math.isfinite(numeric):
        return "--"
    return str(int(round(numeric)))


def tex_escape(text: str) -> str:
    return str(text).replace("_", r"\_").replace("%", r"\%")


def method_sort_key(method: str) -> int:
    try:
        return METHOD_ORDER.index(method)
    except ValueError:
        return len(METHOD_ORDER)


def build_rows(df: pd.DataFrame) -> List[dict[str, str]]:
    df = df[df["method"].astype(str).isin(METHOD_ORDER)].copy()
    if "total_vars" in df.columns and "total_constrs" in df.columns:
        df["_has_model_size"] = (
            pd.to_numeric(df["total_vars"], errors="coerce").notna()
            & pd.to_numeric(df["total_constrs"], errors="coerce").notna()
        )
        df = df[df["_has_model_size"]].copy()

    rows = []
    for method in METHOD_ORDER:
        cand = df[df["method"].astype(str).eq(method)].copy()
        if cand.empty:
            raise RuntimeError(f"No successful model-complexity probe row found for {method}.")
        row = cand.iloc[0]
        rows.append(
            {
                "Method": method,
                "Cont. vars.": fmt_int(row.get("cont_vars")),
                "Bin. vars.": fmt_int(row.get("bin_vars")),
                "Constrs.": fmt_int(row.get("total_constrs")),
            }
        )
    return rows


def render_table(rows: List[dict[str, str]]) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\caption{Optimization Model Complexity on the 119-Bus System}",
        r"\label{tab:model_complexity_119}",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Method & Cont. vars. & Bin. vars. & Constrs. \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{tex_escape(row['Method'])} & "
            f"{row['Cont. vars.']} & "
            f"{row['Bin. vars.']} & "
            f"{row['Constrs.']} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create the 119-bus model-complexity LaTeX table.")
    parser.add_argument("--input", default="formal/step11_119_model_complexity_summary.csv")
    parser.add_argument("--output", default="paper_tables/119/table_119_model_complexity_compact.tex")
    return parser


def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = RESULT_DIR / args.input
    if not input_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_path}. "
            "Run `python -m experiments_v2.summary.probe_119_model_complexity` first."
        )

    df = pd.read_csv(input_path, encoding="utf-8-sig")
    rows = build_rows(df)
    table_text = render_table(rows)

    output_path = RESULT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(table_text, encoding="utf-8")
    print("Saved:", output_path)
    print("")
    print(table_text)


if __name__ == "__main__":
    main()
