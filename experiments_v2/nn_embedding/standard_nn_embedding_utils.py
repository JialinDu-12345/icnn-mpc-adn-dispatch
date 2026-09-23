from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB


Layer = Tuple[np.ndarray, np.ndarray]


def load_standard_critic_layers(path: Path) -> Tuple[List[Layer], List[Layer]]:
    data = np.load(str(path), allow_pickle=False)
    q1_n = int(data["q1_n_layers"][0])
    q2_n = int(data["q2_n_layers"][0])

    q1_layers = [(data[f"q1_W{i}"], data[f"q1_b{i}"]) for i in range(q1_n)]
    q2_layers = [(data[f"q2_W{i}"], data[f"q2_b{i}"]) for i in range(q2_n)]
    return q1_layers, q2_layers


def interval_affine_bounds(
    weight: np.ndarray,
    bias: np.ndarray,
    x_lb: np.ndarray,
    x_ub: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
                                                    

    weight = np.asarray(weight, dtype=float)
    bias = np.asarray(bias, dtype=float)
    x_lb = np.asarray(x_lb, dtype=float)
    x_ub = np.asarray(x_ub, dtype=float)

    weight_pos = np.maximum(weight, 0.0)
    weight_neg = np.minimum(weight, 0.0)
    y_lb = weight_pos @ x_lb + weight_neg @ x_ub + bias
    y_ub = weight_pos @ x_ub + weight_neg @ x_lb + bias
    return y_lb, y_ub


def relu_big_m(
    model: gp.Model,
    pre_var,
    lb: float,
    ub: float,
    name: str,
):
                                                               

                                                                   
       

    if lb >= 0.0:
        y = model.addVar(lb=lb, ub=ub, vtype=GRB.CONTINUOUS, name=f"{name}_relu_pos")
        model.addConstr(y == pre_var, name=f"{name}_relu_pos_eq")
        return y, None

    if ub <= 0.0:
        y = model.addVar(lb=0.0, ub=0.0, vtype=GRB.CONTINUOUS, name=f"{name}_relu_zero")
        return y, None

    y = model.addVar(lb=0.0, ub=max(ub, 0.0), vtype=GRB.CONTINUOUS, name=f"{name}_relu")
    active = model.addVar(vtype=GRB.BINARY, name=f"{name}_relu_bin")

    model.addConstr(y >= pre_var, name=f"{name}_relu_lb_pre")
    model.addConstr(y >= 0.0, name=f"{name}_relu_lb_zero")
    model.addConstr(y <= pre_var - lb * (1.0 - active), name=f"{name}_relu_ub_pre")
    model.addConstr(y <= ub * active, name=f"{name}_relu_ub_zero")
    return y, active


def _linear_expr(weight_row: np.ndarray, variables: Sequence, bias: float) -> gp.LinExpr:
    expr = gp.LinExpr(float(bias))
    for idx, var in enumerate(variables):
        coeff = float(weight_row[idx])
        if coeff != 0.0:
            expr += coeff * var
    return expr


def add_standard_relu_network(
    model: gp.Model,
    x_vars: Sequence,
    layers: List[Layer],
    x_lb: np.ndarray,
    x_ub: np.ndarray,
    name: str,
):
                                                           

                                                                                    
                                                      
       

    current_vars = list(x_vars)
    current_lb = np.asarray(x_lb, dtype=float)
    current_ub = np.asarray(x_ub, dtype=float)

    binary_vars = []
    pre_bounds_log = []

    for layer_idx, (weight, bias) in enumerate(layers):
        weight = np.asarray(weight, dtype=float)
        bias = np.asarray(bias, dtype=float)

        if weight.shape[1] != len(current_vars):
            raise ValueError(
                f"{name} layer {layer_idx} expects {weight.shape[1]} inputs, "
                f"got {len(current_vars)}."
            )

        pre_lb, pre_ub = interval_affine_bounds(weight, bias, current_lb, current_ub)
        pre_bounds_log.append((pre_lb.copy(), pre_ub.copy()))

        pre_vars = []
        for unit_idx in range(weight.shape[0]):
            pre_var = model.addVar(
                lb=float(pre_lb[unit_idx]),
                ub=float(pre_ub[unit_idx]),
                vtype=GRB.CONTINUOUS,
                name=f"{name}_l{layer_idx}_pre{unit_idx}",
            )
            model.addConstr(
                pre_var == _linear_expr(weight[unit_idx], current_vars, float(bias[unit_idx])),
                name=f"{name}_l{layer_idx}_affine{unit_idx}",
            )
            pre_vars.append(pre_var)

        is_last = layer_idx == len(layers) - 1
        if is_last:
            if len(pre_vars) != 1:
                raise ValueError(f"{name} final layer must be scalar, got {len(pre_vars)} outputs.")
            return pre_vars[0], binary_vars, pre_bounds_log

        next_vars = []
        next_lb = np.maximum(pre_lb, 0.0)
        next_ub = np.maximum(pre_ub, 0.0)

        for unit_idx, pre_var in enumerate(pre_vars):
            y, binary_var = relu_big_m(
                model,
                pre_var,
                lb=float(pre_lb[unit_idx]),
                ub=float(pre_ub[unit_idx]),
                name=f"{name}_l{layer_idx}_u{unit_idx}",
            )
            next_vars.append(y)
            if binary_var is not None:
                binary_vars.append(binary_var)

        current_vars = next_vars
        current_lb = next_lb
        current_ub = next_ub

    raise RuntimeError(f"{name} has no layers to embed.")

