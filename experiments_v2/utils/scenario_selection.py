from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Optional

from experiments_v2.config import RESULT_DIR, SPLITS


def dedupe_preserve_order(scenario_ids: List[int]) -> List[int]:
    out: List[int] = []
    seen = set()
    for scenario_id in scenario_ids:
        if scenario_id in seen:
            continue
        out.append(int(scenario_id))
        seen.add(int(scenario_id))
    return out


def parse_scenario_token(token: str) -> List[int]:
    text = token.strip()
    if not text:
        return []
    if ":" not in text:
        return [int(text)]

    start_text, end_text = [part.strip() for part in text.split(":", maxsplit=1)]
    if not start_text or not end_text:
        raise ValueError(f"Scenario range '{token}' must be formatted as start:end.")
    start = int(start_text)
    end = int(end_text)
    step = 1 if end >= start else -1
    return list(range(start, end + step, step))


def parse_csv_scenarios(
    raw: Optional[str],
    *,
    system: str,
    scenario_count: Optional[int] = None,
) -> List[int]:
    if raw is None or raw.strip().lower() == "all":
        scenario_ids = list(SPLITS[system]["test_id"])
    else:
        scenario_ids = []
        for item in raw.split(","):
            scenario_ids.extend(parse_scenario_token(item))

    scenario_ids = dedupe_preserve_order(scenario_ids)
    if scenario_count is not None:
        return scenario_ids[:scenario_count]
    return scenario_ids


def resolve_result_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return RESULT_DIR / path


def load_scenario_file(
    raw_path: str,
    *,
    scenario_count: Optional[int] = None,
) -> List[int]:
    path = resolve_result_path(raw_path)
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if "selected_scenarios" in payload:
            scenario_ids = [int(value) for value in payload["selected_scenarios"]]
        elif "scenario_ids" in payload:
            scenario_ids = [int(value) for value in payload["scenario_ids"]]
        else:
            raise KeyError(
                f"{path} must contain either 'selected_scenarios' or 'scenario_ids'."
            )
    else:
        scenario_ids = []
        with path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "scenario_id" not in reader.fieldnames:
                raise KeyError(f"{path} must contain a 'scenario_id' column.")
            for row in reader:
                text = str(row.get("scenario_id", "")).strip()
                if text:
                    scenario_ids.append(int(float(text)))

    scenario_ids = dedupe_preserve_order(scenario_ids)
    if scenario_count is not None:
        return scenario_ids[:scenario_count]
    return scenario_ids


def resolve_scenarios(
    *,
    system: str,
    scenarios: Optional[str],
    scenario_file: Optional[str],
    scenario_count: Optional[int],
) -> List[int]:
    if scenario_file:
        return load_scenario_file(scenario_file, scenario_count=scenario_count)
    return parse_csv_scenarios(
        scenarios,
        system=system,
        scenario_count=scenario_count,
    )
