
# parser_transistor.py
# Parse a simplified transistor-level JSON (devices/nets/constraints) into a graph-like dict
# compatible with GNN/Transformer pipelines.

import json
import numpy as np
from collections import defaultdict
from typing import Dict, Any

MOS_TYPES = ["NMOS", "PMOS"]

def parse_transistor_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = json.load(f)
    devices = data["devices"]
    nets = data["nets"]
    constraints = data.get("constraints", {})

    name2idx = {d["name"]: i for i, d in enumerate(devices)}

    # Convert multi-pin nets into device-index lists
    net_pin_indices = []
    deg = defaultdict(int)
    for net in nets:
        dev_idxs = []
        for pin in net[:-1]:
            dev_name = pin.split(".")[0]
            if dev_name in name2idx:
                dev_idxs.append(name2idx[dev_name])
        dev_idxs = list(sorted(set(dev_idxs)))
        if len(dev_idxs) >= 2:
            net_pin_indices.append(dev_idxs)
            # degree accumulation (simple proxy)
            for u in dev_idxs:
                deg[u] += len(dev_idxs) - 1

    # Node features: [type_onehot(2), nf, w, l, degree]
    X = []
    for i, d in enumerate(devices):
        t_onehot = [1.0 if d["type"] == t else 0.0 for t in MOS_TYPES]
        feat = t_onehot + [
            float(d.get("nf", 1)),
            float(d.get("w", 1.0)),
            float(d.get("l", 0.05)),
            float(deg[i]),
        ]
        X.append(feat)
    X = np.asarray(X, dtype=np.float32)

    # Simple adjacency: fully connect devices within a net (undirected)
    N = len(devices)
    adj = np.eye(N, dtype=np.float32)
    for lst in net_pin_indices:
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                adj[a, b] = adj[b, a] = 1.0
    # Row-normalize
    row_sums = adj.sum(axis=1, keepdims=True)
    adj = adj / (row_sums + 1e-8)

    # Edge index (PyG-style) – optional
    edge_pairs = []
    for lst in net_pin_indices:
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                edge_pairs.append((a, b)); edge_pairs.append((b, a))
    if edge_pairs:
        edge_index = np.array(edge_pairs, dtype=np.int64).T
    else:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    grid = {
        "row_pitch": float(constraints.get("row_pitch", 1.0)),
        "poly_pitch": float(constraints.get("poly_pitch", 0.1)),
        "y_pmos": 1.0,
        "y_nmos": 0.0,
    }

    return {
        "devices": devices,
        "X": X,
        "adj": adj,
        "edge_index": edge_index,
        "nets_index": net_pin_indices,
        "movable_indices": list(range(N)),
        "pair_map": constraints.get("pair_map", {}),
        "grid": grid,
        "name2idx": name2idx,
    }
