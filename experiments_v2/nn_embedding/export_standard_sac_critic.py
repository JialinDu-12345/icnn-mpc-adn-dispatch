from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch


Layer = Tuple[np.ndarray, np.ndarray]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOLVER_DIR = PROJECT_ROOT / "experiment_assets" / "solvers" / "33" / "std_nn_mpc"


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().astype(float)
    return np.asarray(value, dtype=float)


def extract_state_dict_layers(state_dict: Mapping[str, object]) -> List[Layer]:
                                                                     

                                        
                                         
       

    expected_keys = [
        ("fc1.weight", "fc1.bias"),
        ("fc2.weight", "fc2.bias"),
        ("fc_out.weight", "fc_out.bias"),
    ]
    missing = [key for pair in expected_keys for key in pair if key not in state_dict]
    if missing:
        raise KeyError(f"Missing standard SAC critic keys: {missing}")

    return [(_to_numpy(state_dict[w_key]), _to_numpy(state_dict[b_key])) for w_key, b_key in expected_keys]


def extract_module_layers(module: torch.nn.Module) -> List[Layer]:
    layers: List[Layer] = []
    for layer in module.modules():
        if isinstance(layer, torch.nn.Linear):
            layers.append(
                (
                    layer.weight.detach().cpu().numpy().astype(float),
                    layer.bias.detach().cpu().numpy().astype(float),
                )
            )
    if not layers:
        raise RuntimeError("No Linear layers found in the given critic module.")
    return layers


def load_critic_layers(path: Path) -> List[Layer]:
    obj = torch.load(str(path), map_location="cpu")
    if isinstance(obj, (dict, OrderedDict)):
        return extract_state_dict_layers(obj)
    if isinstance(obj, torch.nn.Module):
        return extract_module_layers(obj)
    raise TypeError(f"Unsupported checkpoint type from {path}: {type(obj)!r}")


def add_layers_to_arrays(arrays: Dict[str, np.ndarray], prefix: str, layers: Iterable[Layer]) -> None:
    layer_list = list(layers)
    arrays[f"{prefix}_n_layers"] = np.array([len(layer_list)], dtype=int)
    for idx, (weight, bias) in enumerate(layer_list):
        arrays[f"{prefix}_W{idx}"] = np.asarray(weight, dtype=float)
        arrays[f"{prefix}_b{idx}"] = np.asarray(bias, dtype=float)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export copied standard SAC critics to a NumPy archive.")
    parser.add_argument(
        "--critic-1",
        default=str(DEFAULT_SOLVER_DIR / "standard_critic_1_model.pth"),
        help="Copied standard SAC critic-1 state_dict.",
    )
    parser.add_argument(
        "--critic-2",
        default=str(DEFAULT_SOLVER_DIR / "standard_critic_2_model.pth"),
        help="Copied standard SAC critic-2 state_dict.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_SOLVER_DIR / "standard_sac_critics_33.npz"),
        help="Output .npz used by the StdNN-H3-MIP solver.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    critic_1_path = Path(args.critic_1)
    critic_2_path = Path(args.critic_2)
    output_path = Path(args.output)

    q1_layers = load_critic_layers(critic_1_path)
    q2_layers = load_critic_layers(critic_2_path)

    arrays: Dict[str, np.ndarray] = {}
    add_layers_to_arrays(arrays, "q1", q1_layers)
    add_layers_to_arrays(arrays, "q2", q2_layers)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **arrays)

    print("Saved:", output_path)
    print("Q1 layers:", [(weight.shape, bias.shape) for weight, bias in q1_layers])
    print("Q2 layers:", [(weight.shape, bias.shape) for weight, bias in q2_layers])


if __name__ == "__main__":
    main()

