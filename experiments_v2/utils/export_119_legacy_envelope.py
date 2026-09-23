from __future__ import annotations

from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from experiments_v2.config import DATA_DIRS, METHOD_DIRS_119              
from experiments_v2.run_119_mpc import (              
    _legacy_bounds_and_network,
    load_legacy_mpc1_definitions,
)


def export_legacy_119_load_envelope() -> Path:
                                                                             

    method_dir = METHOD_DIRS_119["pure_mpc_clean"]
    module = load_legacy_mpc1_definitions(
        module_name="legacy_119_envelope_export",
        file_path=method_dir / "MPC_1.py",
    )
    (
        load_min,
        load_max,
        _list_r_x_pu,
        _list_p_q_pu,
        load_min_Q,
        load_max_Q,
    ) = _legacy_bounds_and_network(module)

    out_dir = DATA_DIRS["119"]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "legacy_119_load_envelope.npz"
    np.savez_compressed(
        out_path,
        load_min=np.asarray(load_min, dtype=float),
        load_max=np.asarray(load_max, dtype=float),
        load_min_Q=np.asarray(load_min_Q, dtype=float),
        load_max_Q=np.asarray(load_max_Q, dtype=float),
    )
    return out_path


def main() -> None:
    out_path = export_legacy_119_load_envelope()
    print(f"Saved 119 legacy load envelope: {out_path}")


if __name__ == "__main__":
    main()
