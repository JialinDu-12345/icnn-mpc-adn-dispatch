from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import pandas as pd

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import RESULT_DIR              


MAIN_COLUMNS = [
    "method",
    "display_forecast_case",
    "horizon",
    "gamma_Q",
    "n_completed",
    "completed_rate",
    "certified_global_rate",
    "mean_cost",
    "ci95_cost",
    "mean_gap_to_perfect_percent",
    "ci95_gap_to_perfect_percent",
    "mean_voltage_violation",
    "ci95_voltage_violation",
    "mean_solve_time_s",
    "ci95_solve_time_s",
    "mean_num_bin_vars",
    "mean_mip_gap",
]

FORECAST_COLUMNS = [
    "forecast_case",
    "baseline",
    "target",
    "n_matched",
    "mean_cost_diff",
    "ci95_cost_diff",
    "mean_cost_reduction_percent",
    "ci95_cost_reduction_percent",
]

NN_COLUMNS = [
    "method",
    "n_perf_rows",
    "completed_rate",
    "mean_cost",
    "ci95_cost",
    "mean_solve_time_s",
    "ci95_solve_time_s",
    "mean_num_vars",
    "mean_num_bin_vars",
    "mean_num_constrs",
    "mean_mip_gap",
]

NN_METHOD_ORDER = ["MPC-3", "StdNN-H3-MIP", "Proposed-H3"]
NN_MODEL_SIZE_FIELDS = ["mean_num_vars", "mean_num_bin_vars", "mean_num_constrs", "mean_mip_gap"]


def readable_number(value: object) -> str:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        text = "" if value is None else str(value)
        return "" if text.lower() == "nan" else text
    if pd.isna(numeric):
        return ""
    if abs(numeric) >= 1000:
        return f"{numeric:.1f}"
    if abs(numeric) >= 100:
        return f"{numeric:.2f}"
    if abs(numeric) >= 10:
        return f"{numeric:.3f}"
    return f"{numeric:.4f}"


def select_existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    existing = [column for column in columns if column in df.columns]
    return df.loc[:, existing].copy()


def format_for_paper(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for column in out.columns:
        if column in {"method", "display_forecast_case", "forecast_case", "baseline", "target"}:
            out[column] = out[column].fillna("").astype(str)
        else:
            out[column] = out[column].map(readable_number)
    return out


def markdown_table(df: pd.DataFrame) -> str:
    headers = list(df.columns)
    rows: List[List[str]] = [[str(value) for value in row] for row in df.to_numpy()]
    widths = [
        max(len(header), *(len(row[idx]) for row in rows)) if rows else len(header)
        for idx, header in enumerate(headers)
    ]
    header_line = "| " + " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |"
    body = [
        "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line, *body]) + "\n"


def latex_escape(value: object) -> str:
    text = str(value)
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
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def latex_table(df: pd.DataFrame) -> str:
    headers = [latex_escape(column) for column in df.columns]
    lines = [
        rf"\begin{{tabular}}{{{'l' * len(headers)}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for row in df.to_numpy():
        lines.append(" & ".join(latex_escape(value) for value in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def write_table(df: pd.DataFrame, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / f"{stem}.csv"
    md_path = output_dir / f"{stem}.md"
    tex_path = output_dir / f"{stem}.tex"

    df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    formatted = format_for_paper(df)
    md_path.write_text(markdown_table(formatted), encoding="utf-8")
    tex_path.write_text(latex_table(formatted), encoding="utf-8")

    print("Saved:", raw_path)
    print("Saved:", md_path)
    print("Saved:", tex_path)


def read_optional(relative_path: str) -> pd.DataFrame | None:
    path = RESULT_DIR / relative_path
    if not path.exists():
        print("[MISS]", path)
        return None
    return pd.read_csv(path, dtype={"gamma_Q": str})


def missing(value: object) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip() == ""


def first_method_row(df: pd.DataFrame | None, method: str) -> pd.Series | None:
    if df is None or "method" not in df.columns:
        return None
    rows = df[df["method"].astype(str).eq(method)].copy()
    if rows.empty:
        return None
    if "n_runs" in rows.columns:
        rows["_n_runs_sort"] = pd.to_numeric(rows["n_runs"], errors="coerce").fillna(-1)
        rows = rows.sort_values("_n_runs_sort", ascending=False)
    return rows.iloc[0]


def combine_nn_summary(
    nn_summary: pd.DataFrame | None,
    main_summary: pd.DataFrame | None,
    stdnn_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    rows = []
    for method in NN_METHOD_ORDER:
        if method == "StdNN-H3-MIP":
            perf_sources = [stdnn_summary, nn_summary, main_summary]
            model_sources = [stdnn_summary, nn_summary, main_summary]
        else:
            perf_sources = [main_summary, nn_summary]
            model_sources = [nn_summary, main_summary]

        perf_row = next(
            (row for row in (first_method_row(df, method) for df in perf_sources) if row is not None),
            None,
        )
        if perf_row is None:
            continue
        merged = perf_row.to_dict()
        model_row = next(
            (row for row in (first_method_row(df, method) for df in model_sources) if row is not None),
            None,
        )
        if model_row is not None:
            for field in NN_MODEL_SIZE_FIELDS:
                if field in model_row and missing(merged.get(field)):
                    merged[field] = model_row[field]
        rows.append(merged)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["method"] = pd.Categorical(out["method"], NN_METHOD_ORDER, ordered=True)
    return out.sort_values("method")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create paper-ready 33-bus tables.")
    parser.add_argument("--main-table", default="formal/step8_33_main_table_with_perfect_gap.csv")
    parser.add_argument("--main-summary", default="formal/step4_33_main_model_based_summary.csv")
    parser.add_argument("--forecast-pairwise", default="formal/step5_33_pairwise_mpc3_vs_proposed_by_case.csv")
    parser.add_argument("--nn-summary", default="formal/step6_33_nn_embedding_summary.csv")
    parser.add_argument("--stdnn-summary", default="formal/step6_33_stdnn_only_summary.csv")
    parser.add_argument("--output-dir", default="paper_tables/33")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    output_dir = RESULT_DIR / args.output_dir

    main_df = read_optional(args.main_table)
    if main_df is not None:
        write_table(select_existing_columns(main_df, MAIN_COLUMNS), output_dir, "table_33_main")

    forecast_df = read_optional(args.forecast_pairwise)
    if forecast_df is not None:
        write_table(
            select_existing_columns(forecast_df, FORECAST_COLUMNS),
            output_dir,
            "table_33_forecast_sensitivity",
        )

    nn_df = combine_nn_summary(
        read_optional(args.nn_summary),
        read_optional(args.main_summary),
        read_optional(args.stdnn_summary),
    )
    if not nn_df.empty:
        write_table(select_existing_columns(nn_df, NN_COLUMNS), output_dir, "table_33_nn_embedding")


if __name__ == "__main__":
    main()
