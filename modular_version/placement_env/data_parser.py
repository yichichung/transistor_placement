"""
Data structures and JSON parsing for transistor placement.
Converts JSON netlist files into graph structures for RL training.
"""
from __future__ import annotations
import json
import pathlib
from typing import List, Dict, Union, Optional
from dataclasses import dataclass

import numpy as np
import torch


# Constants
POWER_NET_NAMES = {"VDD", "VSS", "vdd", "vss", "VDD!", "VSS!"}
MOS_TYPES = ["NMOS", "PMOS"]


@dataclass
class TransistorNode:
    """Represents a transistor device node."""
    name: str
    device_type: str
    width: float
    length: float
    nf: int
    vt: str
    x: Optional[float] = None
    y: Optional[float] = None
    is_pin: bool = False


def _is_power_net(net_name: str) -> bool:
    """Check if a net name is a power/ground net."""
    if not isinstance(net_name, str):
        return False
    return net_name in POWER_NET_NAMES


def _want_pin_token(tok: str) -> bool:
    """
    Filter valid pin tokens (S/D/G only, excluding bulk).
    Only accepts 'DEV.PIN' format.
    """
    if not isinstance(tok, str) or "." not in tok:
        return False
    dev, pin = tok.split(".", 1)
    return pin.upper() in {"S", "D", "G"}


def build_graph_from_nets(devices: List[dict], nets: List[List[str]]) -> torch.Tensor:
    """
    Build adjacency matrix using clique expansion.
    Ignores power nets and bulk pins.

    Args:
        devices: List of device dictionaries
        nets: List of netlist connections

    Returns:
        torch.Tensor: Row-normalized adjacency matrix with self-loops
    """
    name_to_idx = {d["name"]: i for i, d in enumerate(devices)}
    N = len(devices)
    adj_matrix = np.zeros((N, N), dtype=np.float32)

    for net in nets:
        if not net:
            continue
        net_name = net[-1] if isinstance(net[-1], str) else None
        if _is_power_net(net_name):
            continue

        # Only take S/D/G pins, not B (bulk)
        pins = [tok for tok in net[:-1] if _want_pin_token(tok)]
        if len(pins) < 2:
            continue

        # Map to device indices, deduplicate
        dev_indices = sorted({name_to_idx.get(tok.split(".", 1)[0]) for tok in pins
                              if name_to_idx.get(tok.split(".", 1)[0]) is not None})
        if len(dev_indices) < 2:
            continue

        # Clique expansion
        for i in range(len(dev_indices)):
            for j in range(i + 1, len(dev_indices)):
                u, v = dev_indices[i], dev_indices[j]
                if u != v:
                    adj_matrix[u, v] = 1.0
                    adj_matrix[v, u] = 1.0

    # Self-loop + row-normalize
    for i in range(N):
        adj_matrix[i, i] = 1.0
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    adj_matrix = adj_matrix / (row_sums + 1e-8)

    return torch.tensor(adj_matrix, dtype=torch.float32)


def parse_transistor_json(path: Union[str, pathlib.Path], verbose: bool = False) -> dict:
    """
    Parse transistor netlist JSON file into graph structure.

    Args:
        path: Path to JSON file
        verbose: Print debug information

    Returns:
        dict: Graph data containing:
            - devices: Original device list
            - nodes: List of TransistorNode objects
            - features: Node feature matrix (torch.Tensor)
            - adj: Adjacency matrix (torch.Tensor)
            - netlist: List of nets for HPWL calculation
            - movable_indices: Indices of movable devices
            - pair_map: NMOS-PMOS pairing constraints
            - name2idx: Device name to index mapping
            - grid: Layout grid parameters
            - num_cells: Number of devices
            - cell_name: Circuit name
            - pin_nets: Pin connectivity information
    """
    path = pathlib.Path(path)
    with open(path, "r") as f:
        data = json.load(f)

    devices = data["devices"]
    nets = data["nets"]
    constraints = data.get("constraints", {})

    name2idx = {d["name"]: i for i, d in enumerate(devices)}

    # Build TransistorNode objects
    nodes = []
    for d in devices:
        nodes.append(TransistorNode(
            name=d["name"],
            device_type=d["type"],
            width=float(d.get("w", 1.0)),
            length=float(d.get("l", 0.05)),
            nf=int(d.get("nf", 1)),
            vt=d.get("vt", "SVT"),
            is_pin=False
        ))

    # Build adjacency matrix (ignores power/bulk)
    adj = build_graph_from_nets(devices, nets)

    # Pin connectivity: only store S/D/G
    pin_nets = {i: {"S": None, "D": None, "G": None}
                for i in range(len(devices))}

    def set_pin(dev_name: str, pin_letter: str, net_name: str):
        if not dev_name or not pin_letter or not net_name:
            return
        idx = name2idx.get(dev_name)
        if idx is None:
            return
        p = pin_letter.upper()
        if p in ("S", "D", "G"):
            pin_nets[idx][p] = net_name

    for net in nets:
        if not net:
            continue
        net_name = net[-1] if isinstance(net[-1], str) else None
        for token in net[:-1]:
            if isinstance(token, str) and "." in token:
                dev, pin = token.split(".", 1)
                set_pin(dev, pin, net_name)

    # Netlist for HPWL: filter power nets, only use S/D/G pins
    net_pin_indices = []
    for net in nets:
        if not net:
            continue
        net_name = net[-1] if isinstance(net[-1], str) else None
        if _is_power_net(net_name):
            continue

        dev_idxs = []
        for tok in net[:-1]:
            if _want_pin_token(tok):
                dev_name = tok.split(".", 1)[0]
                if dev_name in name2idx:
                    dev_idxs.append(name2idx[dev_name])
        dev_idxs = sorted(set(dev_idxs))
        if len(dev_idxs) >= 1:
            net_pin_indices.append(dev_idxs)

    # Compute degree from filtered adjacency matrix
    N = len(devices)
    deg = {i: int((adj[i].sum().item() - 1.0) * N) for i in range(N)}

    # Build feature matrix
    X = []
    for i, d in enumerate(devices):
        t_onehot = [1.0 if d["type"] == t else 0.0 for t in MOS_TYPES]
        feat = t_onehot + [
            float(d.get("nf", 1)),
            float(d.get("w", 1.0)),
            float(d.get("l", 0.05)),
            float(deg[i])
        ]
        X.append(feat)
    X = np.asarray(X, dtype=np.float32)

    pair_map = constraints.get("pair_map", {})
    grid = {
        "row_pitch": float(constraints.get("row_pitch", 1.0)),
        "poly_pitch": float(constraints.get("poly_pitch", 0.056)),
        "y_pmos": float(constraints.get("y_pmos", 1.0)),
        "y_nmos": float(constraints.get("y_nmos", 0.0))
    }

    if verbose:
        num_edges = int((adj.sum().item() - N) / 2)
        print(
            f"[Graph] Nodes={N}, Nets={len(net_pin_indices)}, Edges={num_edges}")

    return {
        "devices": devices,
        "nodes": nodes,
        "features": torch.tensor(X, dtype=torch.float32),
        "adj": adj,
        "netlist": net_pin_indices,
        "movable_indices": list(range(N)),
        "pair_map": pair_map,
        "name2idx": name2idx,
        "grid": grid,
        "num_cells": N,
        "cell_name": path.stem,
        "pin_nets": pin_nets,
    }
