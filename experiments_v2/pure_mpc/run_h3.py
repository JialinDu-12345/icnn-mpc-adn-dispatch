from __future__ import annotations

import sys
from typing import Iterable, List, Optional

from experiments_v2.comparison.run_33_main import main as run_main


def _force_task(argv: Optional[Iterable[str]], task: str) -> List[str]:
    raw_args = list(sys.argv[1:] if argv is None else argv)
    cleaned: List[str] = []
    skip_next = False
    for item in raw_args:
        if skip_next:
            skip_next = False
            continue
        if item == "--tasks":
            skip_next = True
            continue
        if item.startswith("--tasks="):
            continue
        cleaned.append(item)
    return ["--tasks", task, *cleaned]


def main(argv: Optional[Iterable[str]] = None) -> None:
    run_main(_force_task(argv, "MPC-3"))


if __name__ == "__main__":
    main()
