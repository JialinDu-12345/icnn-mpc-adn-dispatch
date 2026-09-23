from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from experiments_v2.config import RAW_RESULT_COLUMNS, RESULT_DIR, ensure_result_dir


class ResultLogger:
                                                                           

    def __init__(
        self,
        filename: str = "all_raw_results.csv",
        columns: Optional[Iterable[str]] = None,
        result_dir: Path = RESULT_DIR,
    ):
        self.rows: List[Dict[str, Any]] = []
        self.columns = list(columns) if columns is not None else list(RAW_RESULT_COLUMNS)
        self.path = result_dir / filename

    def add(self, **kwargs: Any) -> None:
        self.rows.append(dict(kwargs))

    def add_row(self, row: Dict[str, Any]) -> None:
        self.rows.append(dict(row))

    def _fieldnames(self) -> List[str]:
        extra_columns = []
        known = set(self.columns)
        for row in self.rows:
            for key in row:
                if key not in known and key not in extra_columns:
                    extra_columns.append(key)
        return self.columns + extra_columns

    def save(self) -> Path:
        if not self.rows:
            raise RuntimeError("No result rows to save.")

        ensure_result_dir()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = self._fieldnames()

        with self.path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)

        print(f"Saved results to: {self.path}")
        return self.path


def make_result_row(
    *,
    system: str,
    method: str,
    scenario_id: int,
    split: str,
    forecast_case: str,
    horizon: int,
    seed: int,
    actual_case: str = "id_actual",
    gamma_Q: Optional[float] = None,
    cost: Optional[float] = None,
    perfect_cost: Optional[float] = None,
    gap_percent: Optional[float] = None,
    voltage_violation: Optional[float] = None,
    solve_time_s: Optional[float] = None,
    avg_step_time_s: Optional[float] = None,
    max_step_time_s: Optional[float] = None,
    solver_fail_count: int = 0,
    timeout_count: int = 0,
    completed: Optional[bool] = None,
    incomplete_reason: str = "",
    **extra: Any,
) -> Dict[str, Any]:
    row = {
        "system": system,
        "method": method,
        "scenario_id": int(scenario_id),
        "split": split,
        "actual_case": actual_case,
        "forecast_case": forecast_case,
        "horizon": int(horizon),
        "gamma_Q": gamma_Q,
        "cost": cost,
        "perfect_cost": perfect_cost,
        "gap_percent": gap_percent,
        "voltage_violation": voltage_violation,
        "solve_time_s": solve_time_s,
        "avg_step_time_s": avg_step_time_s,
        "max_step_time_s": max_step_time_s,
        "solver_fail_count": int(solver_fail_count),
        "timeout_count": int(timeout_count),
        "completed": completed,
        "incomplete_reason": incomplete_reason,
        "seed": int(seed),
    }
    row.update(extra)
    return row
