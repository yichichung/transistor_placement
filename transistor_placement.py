"""
Transistor Placement using GNN + Transformer + PPO (SB3)
COMPLETE FIX: Multi-cell training, proper rewards, progress bars, CSV output
Priority: Breaks → Dummy → Shared → HPWL
"""
from __future__ import annotations
import json
import argparse
import pathlib
import csv
import math
import random
import glob
from typing import Optional, Tuple, Dict, List, Any
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gymnasium as gym
except ImportError:
    import gym as gym

from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GNN_output_dim = 128

###############################################################################
# 1. ASAP7 Dataset Parser (FIXED)
###############################################################################

MOS_TYPES = ["NMOS", "PMOS"]


@dataclass
class TransistorNode:
    name: str
    device_type: str
    width: float
    length: float
    nf: int
    vt: str
    x: Optional[float] = None
    y: Optional[float] = None
    is_pin: bool = False


def parse_asap7_cell(asap7_json: pathlib.Path, cell_name: str,
                     poly_pitch: float = 0.056, row_pitch: float = 1.0) -> dict:
    """Parse ASAP7 dataset format and convert to training format"""
    with open(asap7_json, "r") as f:
        data = json.load(f)

    if cell_name not in data:
        available = list(data.keys())[:10]
        raise ValueError(
            f"Cell '{cell_name}' not found. Available (first 10): {available}")

    cell = data[cell_name]
    row_P = cell.get("row_P", [])
    row_N = cell.get("row_N", [])
    prefix = cell.get("prefix", {})

    if prefix.get("poly_pitch") is not None:
        poly_pitch = float(prefix["poly_pitch"])

    devices = []
    name_map_N, name_map_P = {}, {}

    for i, t in enumerate(row_N):
        name = f"MN{i}"
        name_map_N[i] = name
        devices.append({
            "name": name,
            "type": "NMOS",
            "w": float(t.get("W_norm", 1e-7)),
            "l": 0.05,
            "nf": 1,
            "vt": "SVT"
        })

    for i, t in enumerate(row_P):
        name = f"MP{i}"
        name_map_P[i] = name
        devices.append({
            "name": name,
            "type": "PMOS",
            "w": float(t.get("W_norm", 1e-7)),
            "l": 0.05,
            "nf": 1,
            "vt": "SVT"
        })

    net2pins = defaultdict(list)

    def add_connection(dev: str, pin: str, net_name: str):
        if net_name:
            net2pins[net_name].append(f"{dev}.{pin}")

    for i, t in enumerate(row_N):
        nname = name_map_N[i]
        add_connection(nname, "D", t.get("net_drn"))
        add_connection(nname, "S", t.get("net_src"))
        add_connection(nname, "G", t.get("gate_net"))

    for i, t in enumerate(row_P):
        pname = name_map_P[i]
        add_connection(pname, "D", t.get("net_drn"))
        add_connection(pname, "S", t.get("net_src"))
        add_connection(pname, "G", t.get("gate_net"))

    nets = []
    for netname, pins in net2pins.items():
        if len(pins) >= 1:
            nets.append(pins + [netname])

    if len(nets) == 0:
        gate_groups = defaultdict(list)
        for i, t in enumerate(row_N):
            gate = t.get("gate_net")
            if gate:
                gate_groups[gate].append(f"MN{i}")
        for i, t in enumerate(row_P):
            gate = t.get("gate_net")
            if gate:
                gate_groups[gate].append(f"MP{i}")

        for gate_name, dev_names in gate_groups.items():
            if len(dev_names) >= 1:
                pins = [f"{dev}.G" for dev in dev_names]
                nets.append(pins + [gate_name])

    pair_map = {}
    M = min(len(row_N), len(row_P))
    for i in range(M):
        pair_map[name_map_N[i]] = name_map_P[i]

    reverse_pair = {v: k for k, v in pair_map.items()}
    pair_map.update(reverse_pair)

    constraints = {
        "pair_map": pair_map,
        "row_pitch": float(row_pitch),
        "poly_pitch": float(poly_pitch),
        "y_pmos": 1.0,
        "y_nmos": 0.0
    }

    return {
        "devices": devices,
        "nets": nets,
        "constraints": constraints
    }


def build_graph_from_nets(devices: List[dict], nets: List[List[str]], allow_power: bool = True):
    """Build adjacency matrix from nets using clique expansion"""
    name_to_idx = {d["name"]: i for i, d in enumerate(devices)}
    N = len(devices)
    adj_matrix = np.zeros((N, N), dtype=np.float32)
    edges = set()

    def pin_to_dev(pin: str) -> Optional[int]:
        if not isinstance(pin, str) or "." not in pin:
            return None
        dev_name = pin.split(".")[0]
        return name_to_idx.get(dev_name, None)

    nets_processed = 0
    edges_added = 0

    for net_idx, net in enumerate(nets):
        pins = [p for p in net[:-1] if isinstance(p, str) and "." in p]
        if not pins:
            continue

        if not allow_power:
            net_name = net[-1] if len(net) > 0 else ""
            if net_name.upper() in ["VDD", "VSS", "VDDX", "VSSX"]:
                power_pins = sum(1 for p in pins if any(
                    x in p.upper() for x in ["VDD", "VSS"]))
                if power_pins == len(pins):
                    continue

        dev_indices = []
        for pin in pins:
            idx = pin_to_dev(pin)
            if idx is not None:
                dev_indices.append(idx)

        dev_indices = sorted(set(dev_indices))

        if len(dev_indices) < 2:
            continue

        nets_processed += 1
        for i in range(len(dev_indices)):
            for j in range(i + 1, len(dev_indices)):
                u, v = dev_indices[i], dev_indices[j]
                if u != v:
                    edges.add((min(u, v), max(u, v)))
                    adj_matrix[u, v] = 1.0
                    adj_matrix[v, u] = 1.0
                    edges_added += 1

    for i in range(N):
        adj_matrix[i, i] = 1.0

    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    adj_matrix = adj_matrix / (row_sums + 1e-8)

    adj = torch.tensor(adj_matrix, dtype=torch.float32)
    edge_list = sorted(list(edges))

    return adj, edge_list


def parse_transistor_json(path: pathlib.Path, verbose: bool = False) -> dict:
    """Parse environment JSON"""
    with open(path, "r") as f:
        data = json.load(f)

    devices = data["devices"]
    nets = data["nets"]
    constraints = data.get("constraints", {})

    name2idx = {d["name"]: i for i, d in enumerate(devices)}

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

    adj, edge_list = build_graph_from_nets(devices, nets, allow_power=True)

    net_pin_indices = []
    for net in nets:
        dev_idxs = []
        for pin in net[:-1]:
            if isinstance(pin, str) and "." in pin:
                dev_name = pin.split(".")[0]
                if dev_name in name2idx:
                    dev_idxs.append(name2idx[dev_name])

        dev_idxs = list(sorted(set(dev_idxs)))
        if len(dev_idxs) >= 1:
            net_pin_indices.append(dev_idxs)

    N = len(devices)
    deg = {}
    for i in range(N):
        deg[i] = int((adj[i].sum().item() - 1.0) * N)

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

    num_edges = len(edge_list)
    avg_degree = (2 * num_edges) / max(N, 1)

    if verbose:
        print(
            f"[Graph] Nodes={N}, Nets={len(net_pin_indices)}, Edges={num_edges}, AvgDeg={avg_degree:.2f}")

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
        "cell_name": path.stem
    }

###############################################################################
# 2. GNN Encoder
###############################################################################


class GCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(torch.matmul(adj, x)))


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 64, out_dim: int = GNN_output_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hid_dim)
        self.gcn2 = GCNLayer(hid_dim, hid_dim)
        self.gcn3 = GCNLayer(hid_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        h = self.gcn3(h, adj)
        return h

###############################################################################
# 3. Transformer Policy
###############################################################################


class PositionEncoder1D(nn.Module):
    def __init__(self, embed_dim: int, method: str = "fourier", num_bands: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.method = method
        self.num_bands = num_bands

        if method == "fourier":
            in_dim = 1 + 2 * num_bands
        else:
            in_dim = 1

        self.proj = nn.Linear(in_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, col_indices: torch.Tensor) -> torch.Tensor:
        if col_indices is None or col_indices.numel() == 0:
            return None

        if col_indices.dim() == 1:
            col_indices = col_indices.unsqueeze(-1)

        if self.method == "fourier":
            freqs = col_indices.new_tensor(
                [2.0**k for k in range(self.num_bands)]) * (2 * math.pi)
            col_w = col_indices * freqs
            feats = torch.cat(
                [col_indices, torch.sin(col_w), torch.cos(col_w)], dim=-1)
            out = self.proj(feats)
        else:
            out = self.proj(col_indices)

        return self.ln(out)


class CrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim), nn.ReLU(), nn.Linear(ff_dim, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_out, _ = self.mha(
            q, kv, kv, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(q + attn_out)
        ff_out = self.ff(x)
        out = self.ln2(x + ff_out)
        return out


class TransformerPolicy(nn.Module):
    def __init__(self, embed_dim=GNN_output_dim, num_heads=4, ff_dim=None,
                 num_layers=3, enable_bias=True, gamma=2.0, sigma=0.15):
        super().__init__()
        self.embed_dim = embed_dim
        ff_dim = ff_dim or int(embed_dim * 4)

        self.pos1d = PositionEncoder1D(embed_dim=embed_dim)
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, ff_dim)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(embed_dim, 1)

        self.enable_bias = enable_bias
        self.gamma = nn.Parameter(torch.tensor(
            float(gamma)), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(
            float(sigma)), requires_grad=False)

    def _build_bias_1d(self, next_cols: torch.Tensor, hist_cols: torch.Tensor,
                       N: int, device, dtype):
        if not self.enable_bias or next_cols is None or hist_cols is None:
            return None

        T = hist_cols.size(0)
        if T == 0:
            return None

        attn_mask = torch.zeros((N, T), device=device, dtype=dtype)
        diffs = torch.abs(hist_cols.unsqueeze(0) - next_cols.unsqueeze(1))

        denom = 2.0 * max(float(self.sigma.item()), 1e-6) ** 2
        b = -float(self.gamma.item()) * torch.exp(-(diffs * diffs) / denom)
        b = torch.clamp(b, min=-8.0, max=0.0)

        attn_mask[:, :] = b
        return attn_mask

    def forward(self, cell_embeddings: torch.Tensor,
                placed_indices: Optional[set | list | np.ndarray] = None,
                node_positions: Optional[Dict[int, Tuple]] = None,
                next_cols: Optional[torch.Tensor] = None) -> torch.Tensor:
        device = cell_embeddings.device
        dtype = cell_embeddings.dtype
        N = cell_embeddings.size(0)

        if placed_indices is None:
            has_placed = False
        elif isinstance(placed_indices, (set, list)):
            has_placed = len(placed_indices) > 0
        elif isinstance(placed_indices, np.ndarray):
            has_placed = placed_indices.size > 0
        else:
            has_placed = False

        if has_placed:
            if isinstance(placed_indices, set):
                kv_indices = sorted(list(placed_indices))
            elif isinstance(placed_indices, list):
                kv_indices = placed_indices
            else:
                kv_indices = placed_indices.tolist()

            idx_t = torch.tensor(kv_indices, device=device, dtype=torch.long)
            context = cell_embeddings[idx_t].unsqueeze(0)

            hist_cols_list = []
            for i in kv_indices:
                if node_positions and i in node_positions:
                    pos = node_positions[i]
                    if len(pos) >= 5 and pos[4] is not None:
                        hist_cols_list.append(float(pos[4]))
                    else:
                        hist_cols_list.append(0.0)
                else:
                    hist_cols_list.append(0.0)

            if hist_cols_list:
                hist_cols = torch.tensor(
                    hist_cols_list, device=device, dtype=dtype)
                pe1d = self.pos1d(hist_cols)
                context = context.squeeze(0) + pe1d
                context = F.layer_norm(context, (self.embed_dim,)).unsqueeze(0)

                attn_mask = self._build_bias_1d(
                    next_cols, hist_cols, N, device, dtype)
            else:
                attn_mask = None
        else:
            context = cell_embeddings.mean(dim=0, keepdim=True).unsqueeze(0)
            attn_mask = None

        queries = cell_embeddings.unsqueeze(0)
        x = queries
        for blk in self.layers:
            x = blk(x, context, attn_mask=attn_mask)

        scores = self.output_proj(x).squeeze(0).squeeze(-1)
        return scores

###############################################################################
# 4. Value Network
###############################################################################


class ValueNetwork(nn.Module):
    def __init__(self, embed_dim=GNN_output_dim, extra_dim=5, hidden_dim=256):
        super().__init__()
        in_dim = embed_dim + extra_dim
        self.value_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state_embedding: torch.Tensor) -> torch.Tensor:
        return self.value_net(state_embedding).squeeze(-1)

###############################################################################
# 5. Multi-Cell Environment (FIXED with proper rewards)
###############################################################################


class TransistorPlacementEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, graph_data: Dict, reward_cfg: Dict, device=None):
        super().__init__()
        self._rebuild_from_dict(graph_data)
        self.device_tensor = device or torch.device("cpu")
        self.reward_cfg = reward_cfg

        self.observation_space = gym.spaces.Box(
            low=-1, high=self.N, shape=(self.N,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.N)

    def _rebuild_from_dict(self, graph_data: Dict):
        """Rebuild environment from graph data"""
        self.graph = graph_data
        self.nodes: List[TransistorNode] = graph_data["nodes"]
        self.N = len(self.nodes)
        self.devices = graph_data["devices"]
        self.pair_map = graph_data["pair_map"]
        self.grid = graph_data["grid"]
        self.netlist = graph_data["netlist"]

        self.col_idx = 0
        self.pmos_cols = []
        self.nmos_cols = []
        self.placed = np.zeros(self.N, dtype=np.int32)
        self.positions = {}
        self.sequence = []

        self.cur_episode_metrics = {}
        self.episode_steps = 0
        self._last_metrics = None
        self.last_reward = 0.0
        self.current_placement = []

    def _pair_of(self, idx: int) -> Optional[int]:
        name = self.devices[idx]["name"]
        peer_name = self.pair_map.get(name)
        if peer_name is None:
            return None
        return self.graph["name2idx"].get(peer_name)

    def _can_share(self, a_idx: Optional[int], b_idx: Optional[int], row_type: str) -> bool:
        if a_idx is None or b_idx is None:
            return False
        a = self.devices[a_idx]
        b = self.devices[b_idx]
        if a["type"] != row_type or b["type"] != row_type:
            return False
        return True

    def _shared_amount(self, a_idx: Optional[int], b_idx: Optional[int]) -> float:
        if a_idx is None or b_idx is None:
            return 0.0
        a = self.devices[a_idx]
        b = self.devices[b_idx]
        return float(min(a.get("nf", 1), b.get("nf", 1)))

    def _count_breaks_and_shared(self, cols: List, row_type: str) -> Tuple[int, float]:
        breaks, shared = 0, 0.0
        for i in range(len(cols) - 1):
            if self._can_share(cols[i], cols[i+1], row_type):
                shared += self._shared_amount(cols[i], cols[i+1])
            else:
                if cols[i] is not None and cols[i+1] is not None:
                    breaks += 1
        return breaks, shared

    def _estimate_dummy(self) -> Tuple[int, int]:
        dummy = 0
        max_cols = max(len(self.pmos_cols), len(self.nmos_cols))
        for i in range(max_cols):
            p = self.pmos_cols[i] if i < len(self.pmos_cols) else None
            n = self.nmos_cols[i] if i < len(self.nmos_cols) else None
            if (p is None) ^ (n is None):
                dummy += 1
        return dummy, 0

    def _compute_hpwl(self) -> float:
        total = 0.0
        cell_w = self.grid["poly_pitch"] * \
            max(len(self.pmos_cols), len(self.nmos_cols))
        cell_h = self.grid["row_pitch"] * 2
        norm_factor = cell_w + cell_h + 1e-6

        for net in self.netlist:
            xs, ys = [], []
            for idx in net:
                if idx in self.positions:
                    x, y = self.positions[idx][:2]
                    xs.append(x)
                    ys.append(y)
            if len(xs) >= 2:
                total += ((max(xs) - min(xs)) +
                          (max(ys) - min(ys))) / norm_factor
        return total

    def _col_distance_cost(self) -> float:
        name2idx = self.graph["name2idx"]
        total = 0.0
        count = 0
        for p_name, n_name in self.pair_map.items():
            if p_name in name2idx and n_name in name2idx:
                p_idx = name2idx[p_name]
                n_idx = name2idx[n_name]
                if p_idx in self.positions and n_idx in self.positions:
                    p_col = self.positions[p_idx][4] if len(
                        self.positions[p_idx]) > 4 else None
                    n_col = self.positions[n_idx][4] if len(
                        self.positions[n_idx]) > 4 else None
                    if isinstance(p_col, int) and isinstance(n_col, int):
                        total += abs(p_col - n_col)
                        count += 1
        return float(total) / max(count, 1)

    def _metrics(self) -> Dict:
        b_p, s_p = self._count_breaks_and_shared(self.pmos_cols, "PMOS")
        b_n, s_n = self._count_breaks_and_shared(self.nmos_cols, "NMOS")
        dummy, eff_dummy = self._estimate_dummy()
        hpwl = self._compute_hpwl()
        col_dist = self._col_distance_cost()

        return {
            "breaks": b_p + b_n,
            "shared": s_p + s_n,
            "dummy": dummy,
            "eff_dummy": eff_dummy,
            "hpwl": hpwl,
            "col_dist": col_dist
        }

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(self.N, dtype=np.float32)
        for i in range(self.N):
            if self.placed[i] == 1:
                mask[i] = 0.0
        return mask

    def get_current_placement_dicts(self) -> List[Dict]:
        """Get current placement as list of dicts for CSV export"""
        placement = []
        for idx, pos in self.positions.items():
            if idx >= len(self.devices):
                continue

            dev = self.devices[idx]
            x, y, orient, row, col = pos

            pair_idx = self._pair_of(idx)
            pair_name = ""
            if pair_idx is not None and pair_idx < len(self.devices):
                pair_name = self.devices[pair_idx]["name"]

            placement.append({
                "device_name": dev["name"],
                "device_type": dev["type"],
                "row": row,
                "column": col,
                "x": f"{x:.4f}",
                "y": f"{y:.4f}",
                "orient": orient,
                "w": dev.get("w", 1.0),
                "l": dev.get("l", 0.05),
                "nf": dev.get("nf", 1),
                "pair_with": pair_name
            })
        return placement

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.col_idx = 0
        self.pmos_cols, self.nmos_cols = [], []
        self.placed = np.zeros(self.N, dtype=np.int32)
        self.positions = {}
        self.sequence = []
        self.episode_steps = 0
        self.cur_episode_metrics = {}
        self._last_metrics = None
        self.last_reward = 0.0

        obs = np.zeros(self.N, dtype=np.float32)
        info = {"mask": self.get_action_mask()}
        return obs, info

    def step(self, action: int):
        done = False
        info = {}

        if not (0 <= action < self.N):
            obs = np.zeros(self.N, dtype=np.float32)
            return obs, -10.0, False, False, {"invalid_action": True}

        if self.placed[action] == 1:
            obs = np.zeros(self.N, dtype=np.float32)
            for i, idx in enumerate(self.sequence):
                obs[idx] = float(i + 1)
            info["invalid_action"] = True
            info["mask"] = self.get_action_mask()
            return obs, -5.0, False, False, info

        def _place_to_col(idx: int, row_type: str, col: int):
            x = self.grid["poly_pitch"] * col
            y = self.grid["y_pmos"] if row_type == "PMOS" else self.grid["y_nmos"]
            orient = "MX" if row_type == "PMOS" else "R0"
            self.positions[idx] = (x, y, orient, row_type, col)
            self.placed[idx] = 1
            self.sequence.append(idx)

        if self.col_idx >= len(self.pmos_cols):
            self.pmos_cols.append(None)
            self.nmos_cols.append(None)

        dev = self.devices[action]
        row = dev["type"]
        if row == "PMOS":
            self.pmos_cols[self.col_idx] = action
            _place_to_col(action, "PMOS", self.col_idx)
        else:
            self.nmos_cols[self.col_idx] = action
            _place_to_col(action, "NMOS", self.col_idx)

        peer_name = self.pair_map.get(dev["name"])
        if peer_name is not None:
            peer_idx = self.graph["name2idx"].get(peer_name)
            if peer_idx is not None and self.placed[peer_idx] == 0:
                peer_row = self.devices[peer_idx]["type"]
                if peer_row == "PMOS":
                    self.pmos_cols[self.col_idx] = peer_idx
                    _place_to_col(peer_idx, "PMOS", self.col_idx)
                else:
                    self.nmos_cols[self.col_idx] = peer_idx
                    _place_to_col(peer_idx, "NMOS", self.col_idx)

        self.col_idx += 1
        self.episode_steps += 1

        cur = self._metrics()
        if self._last_metrics is None:
            delta = {k: 0.0 for k in cur.keys()}
        else:
            delta = {k: cur[k] - self._last_metrics[k] for k in cur.keys()}
        self._last_metrics = cur
        self.cur_episode_metrics[self.episode_steps] = cur

        # Priority: Breaks >> Dummy >> Shared >> HPWL
        w_break = float(self.reward_cfg.get("w_break", 100.0))
        w_dummy = float(self.reward_cfg.get("w_dummy", 50.0))
        w_share = float(self.reward_cfg.get("w_share", 10.0))
        w_hpwl = float(self.reward_cfg.get("w_hpwl", 2.0))
        w_cdist = float(self.reward_cfg.get("w_cdist", 5.0))

        reward = (
            -w_break * delta["breaks"] +
            -w_dummy * delta["dummy"] +
            w_share * delta["shared"] -
            w_hpwl * delta["hpwl"] -
            w_cdist * delta["col_dist"]
        )

        done = bool(np.all(self.placed == 1))
        if done:
            final = cur
            # Final bonus/penalty
            reward += (
                -w_break * final["breaks"] +
                -w_dummy * final["dummy"] +
                w_share * final["shared"] -
                w_hpwl * final["hpwl"] -
                w_cdist * final["col_dist"]
            )
            info["final_metrics"] = final
            info["episode_steps"] = self.episode_steps
            self.current_placement = self.get_current_placement_dicts()
            self.last_reward = reward

        obs = np.zeros(self.N, dtype=np.float32)
        for i, idx in enumerate(self.sequence):
            obs[idx] = float(i + 1)

        info["mask"] = self.get_action_mask()
        return obs, float(reward), bool(done), False, info


class RandomMultiCellEnv(TransistorPlacementEnv):
    """Environment that randomly selects a cell for each episode"""

    def __init__(self, env_files: List[pathlib.Path], reward_cfg: Dict, device=None):
        self.env_files = [pathlib.Path(p) for p in env_files]
        self.reward_cfg = reward_cfg
        self.device_tensor = device or torch.device("cpu")

        # Load first cell to initialize
        first_data = parse_transistor_json(self.env_files[0], verbose=False)
        super().__init__(first_data, reward_cfg, device)

        print(f"[Multi-Cell] Loaded {len(self.env_files)} cells for training")
        for f in self.env_files[:5]:
            print(f"  - {f.stem}")
        if len(self.env_files) > 5:
            print(f"  ... and {len(self.env_files) - 5} more")

    def reset(self, *, seed=None, options=None):
        # Randomly select a cell
        selected_file = random.choice(self.env_files)
        graph_data = parse_transistor_json(selected_file, verbose=False)

        # Rebuild environment with new cell
        self._rebuild_from_dict(graph_data)

        # Reset action/observation spaces for new size
        self.observation_space = gym.spaces.Box(
            low=-1, high=self.N, shape=(self.N,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.N)

        return super().reset(seed=seed, options=options)

###############################################################################
# 6. SB3 Policy Integration (FIXED)
###############################################################################


class TransistorPolicySB3(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 encoder, policy_net, value_net, graph_data, env_ref, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.encoder = encoder
        self.policy_net = policy_net
        self.value_net_model = value_net
        self.graph = graph_data
        self.env_ref = env_ref

        self.add_module("gnn_encoder", self.encoder)
        self.add_module("transformer_policy", self.policy_net)
        self.add_module("value_network", self.value_net_model)

        self.mlp_extractor = nn.Identity()
        self.action_net = nn.Identity()
        self.value_net = nn.Identity()

    def _compute_embeddings(self):
        device = next(self.encoder.parameters()).device
        features = self.env_ref.graph["features"].to(device)
        adj = self.env_ref.graph["adj"].to(device)
        return self.encoder(features, adj)

    def _build_next_cols(self, embeddings):
        N = embeddings.size(0)
        col_idx = self.env_ref.col_idx
        next_cols = torch.full((N,), float(col_idx),
                               device=embeddings.device, dtype=embeddings.dtype)
        return next_cols

    def _get_state_embedding(self, embeddings):
        placed_indices = np.where(self.env_ref.placed == 1)[0]
        if len(placed_indices) > 0:
            placed_t = torch.tensor(
                placed_indices, device=embeddings.device, dtype=torch.long)
            pooled = embeddings[placed_t].mean(dim=0)
        else:
            pooled = embeddings.mean(dim=0)

        placed_ratio = float(np.sum(self.env_ref.placed)
                             ) / max(self.env_ref.N, 1)
        col_ratio = self.env_ref.col_idx / max(self.env_ref.N, 1)

        m = self.env_ref._metrics()
        scalars = torch.tensor([
            placed_ratio,
            col_ratio,
            float(m["breaks"]),
            float(m["shared"]),
            float(m["dummy"])
        ], device=embeddings.device, dtype=embeddings.dtype)

        return torch.cat([pooled, scalars], dim=-1)

    def forward(self, obs, deterministic=False):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        embeddings = self._compute_embeddings()
        next_cols = self._build_next_cols(embeddings)

        actions_list, values_list, logprob_list = [], [], []

        for i in range(obs.size(0)):
            placed_set = set(np.where(self.env_ref.placed == 1)[0].tolist())

            scores = self.policy_net(
                embeddings,
                placed_indices=placed_set,
                node_positions=self.env_ref.positions,
                next_cols=next_cols
            )

            mask = torch.ones_like(scores, dtype=torch.bool)
            for idx in range(self.env_ref.N):
                if self.env_ref.placed[idx] == 1:
                    mask[idx] = False
            scores = scores.masked_fill(~mask, -1e9)

            dist = torch.distributions.Categorical(logits=scores)
            action = dist.probs.argmax(
                dim=-1) if deterministic else dist.sample()
            log_p = dist.log_prob(action)

            state_embed = self._get_state_embedding(embeddings)
            value = self.value_net_model(state_embed)

            actions_list.append(action)
            values_list.append(value)
            logprob_list.append(log_p)

        actions = torch.stack(actions_list)
        values = torch.stack(values_list)
        log_prob = torch.stack(logprob_list)

        return actions.view(-1), values.view(-1), log_prob.view(-1)

    def _predict(self, observation, deterministic=False):
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions.view(-1)

    def evaluate_actions(self, obs, actions):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        embeddings = self._compute_embeddings()
        next_cols = self._build_next_cols(embeddings)

        log_probs, entropies, values = [], [], []

        for i in range(obs.size(0)):
            placed_set = set(np.where(self.env_ref.placed == 1)[0].tolist())

            scores = self.policy_net(
                embeddings,
                placed_indices=placed_set,
                node_positions=self.env_ref.positions,
                next_cols=next_cols
            )

            mask = torch.ones_like(scores, dtype=torch.bool)
            for idx in range(self.env_ref.N):
                if self.env_ref.placed[idx] == 1:
                    mask[idx] = False
            scores = scores.masked_fill(~mask, -1e9)

            dist = torch.distributions.Categorical(logits=scores)
            log_probs.append(dist.log_prob(actions[i]))
            entropies.append(dist.entropy())

            state_embed = self._get_state_embedding(embeddings)
            values.append(self.value_net_model(state_embed))

        return torch.stack(values), torch.stack(log_probs), torch.stack(entropies)

    def predict_values(self, obs):
        embeddings = self._compute_embeddings()
        if obs.dim() == 1:
            state_embed = self._get_state_embedding(embeddings)
            return self.value_net_model(state_embed).unsqueeze(0)
        else:
            vals = []
            for i in range(obs.size(0)):
                state_embed = self._get_state_embedding(embeddings)
                vals.append(self.value_net_model(state_embed))
            return torch.stack(vals)

###############################################################################
# 7. Callbacks (TQDM + Best Placement)
###############################################################################


class TqdmCallback(BaseCallback):
    """Progress bar with metrics display"""

    def __init__(self, total_timesteps: int, verbose: int = 0):
        super().__init__(verbose)
        self.total = total_timesteps
        self.pbar = None
        self._last_update = 0

    def _on_training_start(self) -> None:
        self.pbar = tqdm(total=self.total, desc="Training", unit=" steps")

    def _on_step(self) -> bool:
        current = self.num_timesteps
        delta = current - self._last_update
        self.pbar.update(delta)
        self._last_update = current
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


class BestPlacementCallback(BaseCallback):
    """Save best placement to CSV"""

    def __init__(self, env, csv_path: pathlib.Path, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.csv_path = csv_path
        self.best_breaks = float('inf')
        self.best_dummy = float('inf')
        self.best_shared = 0.0
        self.best_hpwl = float('inf')
        self.best_score = -float('inf')
        self.header_written = False
        self.episode_count = 0

    def _on_training_start(self) -> None:
        """Write initial placement"""
        try:
            env_single = self.env.envs[0]
            placement = env_single.get_current_placement_dicts()
            if placement:
                self._write_csv(placement)
        except Exception as e:
            if self.verbose:
                print(f"[Warning] Could not write initial placement: {e}")

    def _on_step(self) -> bool:
        try:
            dones = self.locals.get("dones", [False])
            if not any(dones):
                return True

            env_single = self.env.envs[0]
            infos = self.locals.get("infos", [{}])

            if not infos or "final_metrics" not in infos[0]:
                return True

            metrics = infos[0]["final_metrics"]
            self.episode_count += 1

            breaks = metrics["breaks"]
            dummy = metrics["dummy"]
            shared = metrics["shared"]
            hpwl = metrics["hpwl"]
            col_dist = metrics.get("col_dist", 0.0)

            # Priority: Breaks > Dummy > Shared > HPWL
            is_better = False
            if breaks < self.best_breaks:
                is_better = True
            elif breaks == self.best_breaks:
                if dummy < self.best_dummy:
                    is_better = True
                elif dummy == self.best_dummy:
                    if shared > self.best_shared:
                        is_better = True
                    elif shared == self.best_shared and hpwl < self.best_hpwl:
                        is_better = True

            if is_better:
                self.best_breaks = breaks
                self.best_dummy = dummy
                self.best_shared = shared
                self.best_hpwl = hpwl

                placement = env_single.get_current_placement_dicts()
                if placement:
                    self._write_csv(placement)
                    if self.verbose:
                        print(f"\n[Best #{self.episode_count}] Breaks={breaks}, Dummy={dummy}, "
                              f"Shared={shared:.1f}, HPWL={hpwl:.3f}, ColDist={col_dist:.2f}")

        except Exception as e:
            if self.verbose > 1:
                print(f"[Warning] Callback error: {e}")

        return True

    def _on_training_end(self) -> None:
        """Ensure at least one placement is written"""
        if not self.header_written:
            try:
                env_single = self.env.envs[0]
                placement = env_single.get_current_placement_dicts()
                if placement:
                    self._write_csv(placement)
            except Exception as e:
                if self.verbose:
                    print(f"[Warning] Could not write final placement: {e}")

    def _write_csv(self, placement: List[Dict]):
        """Write placement to CSV"""
        try:
            with open(self.csv_path, "w", newline="") as f:
                fieldnames = ["device_name", "device_type", "row", "column",
                              "x", "y", "orient", "w", "l", "nf", "pair_with"]
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(placement)
            self.header_written = True
        except Exception as e:
            if self.verbose:
                print(f"[Error] Failed to write CSV: {e}")

###############################################################################
# 8. Training Pipeline
###############################################################################


def make_sb3_model(graph_data, reward_cfg, device, ppo_kwargs=None):
    feature_dim = graph_data["features"].size(1)
    encoder = GNNEncoder(in_dim=feature_dim, out_dim=GNN_output_dim).to(device)
    policy_net = TransformerPolicy(embed_dim=GNN_output_dim).to(device)
    value_net = ValueNetwork(embed_dim=GNN_output_dim, extra_dim=5).to(device)

    def _make_env():
        return TransistorPlacementEnv(graph_data, reward_cfg, device=device)

    env = DummyVecEnv([_make_env])
    env_ref = env.envs[0]

    policy_kwargs = dict(
        encoder=encoder,
        policy_net=policy_net,
        value_net=value_net,
        graph_data=graph_data,
        env_ref=env_ref
    )

    if ppo_kwargs is None:
        ppo_kwargs = {}

    model = PPO(
        policy=TransistorPolicySB3,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        n_steps=ppo_kwargs.get("n_steps", 2048),
        batch_size=ppo_kwargs.get("batch_size", 64),
        learning_rate=ppo_kwargs.get("learning_rate", 3e-4),
        gamma=ppo_kwargs.get("gamma", 0.99),
        gae_lambda=ppo_kwargs.get("gae_lambda", 0.95),
        clip_range=ppo_kwargs.get("clip_range", 0.2),
        ent_coef=ppo_kwargs.get("ent_coef", 0.02),
        vf_coef=ppo_kwargs.get("vf_coef", 0.5),
        max_grad_norm=ppo_kwargs.get("max_grad_norm", 0.5),
        tensorboard_log=ppo_kwargs.get("tensorboard_log", None)
    )

    return model, env, (encoder, policy_net, value_net)


def make_multicell_model(env_files, reward_cfg, device, ppo_kwargs=None):
    """Create model with multi-cell environment"""

    # Load first cell to get feature dimensions
    first_data = parse_transistor_json(env_files[0], verbose=False)
    feature_dim = first_data["features"].size(1)

    encoder = GNNEncoder(in_dim=feature_dim, out_dim=GNN_output_dim).to(device)
    policy_net = TransformerPolicy(embed_dim=GNN_output_dim).to(device)
    value_net = ValueNetwork(embed_dim=GNN_output_dim, extra_dim=5).to(device)

    def _make_env():
        return RandomMultiCellEnv(env_files, reward_cfg, device=device)

    env = DummyVecEnv([_make_env])
    env_ref = env.envs[0]

    policy_kwargs = dict(
        encoder=encoder,
        policy_net=policy_net,
        value_net=value_net,
        graph_data=first_data,
        env_ref=env_ref
    )

    if ppo_kwargs is None:
        ppo_kwargs = {}

    model = PPO(
        policy=TransistorPolicySB3,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        device=device,
        n_steps=ppo_kwargs.get("n_steps", 2048),
        batch_size=ppo_kwargs.get("batch_size", 64),
        learning_rate=ppo_kwargs.get("learning_rate", 3e-4),
        gamma=ppo_kwargs.get("gamma", 0.99),
        gae_lambda=ppo_kwargs.get("gae_lambda", 0.95),
        clip_range=ppo_kwargs.get("clip_range", 0.2),
        ent_coef=ppo_kwargs.get("ent_coef", 0.02),
        vf_coef=ppo_kwargs.get("vf_coef", 0.5),
        max_grad_norm=ppo_kwargs.get("max_grad_norm", 0.5),
        tensorboard_log=ppo_kwargs.get("tensorboard_log", None)
    )

    return model, env, (encoder, policy_net, value_net)


def train_transistor_placement(
    json_path: Optional[pathlib.Path] = None,
    env_dir: Optional[pathlib.Path] = None,
    output_dir: pathlib.Path = pathlib.Path("./output"),
    total_timesteps: int = 100_000,
    reward_cfg: Optional[Dict] = None,
    ppo_kwargs: Optional[Dict] = None,
    device_arg=None
):
    device_final = device_arg or device
    print(f"Using device: {device_final}")
    if device_final.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA version: {torch.version.cuda}")

    if reward_cfg is None:
        # Priority: Breaks >> Dummy >> Shared >> HPWL
        reward_cfg = {
            "w_break": 100.0,   # Highest priority
            "w_dummy": 50.0,    # Second priority
            "w_share": 10.0,    # Third priority
            "w_hpwl": 2.0,      # Fourth priority
            "w_cdist": 5.0
        }

    print(f"\n[Reward Weights] Priority: Breaks > Dummy > Shared > HPWL")
    print(f"  Breaks: {reward_cfg['w_break']}")
    print(f"  Dummy: {reward_cfg['w_dummy']}")
    print(f"  Shared: {reward_cfg['w_share']}")
    print(f"  HPWL: {reward_cfg['w_hpwl']}")
    print(f"  ColDist: {reward_cfg['w_cdist']}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup tensorboard
    if ppo_kwargs is None:
        ppo_kwargs = {}
    ppo_kwargs["tensorboard_log"] = str(output_dir / "tb")

    # Multi-cell or single-cell mode
    if env_dir is not None:
        print(f"\n[Multi-Cell Mode] Loading cells from: {env_dir}")
        env_files = sorted(glob.glob(str(env_dir / "*.json")))
        if len(env_files) == 0:
            raise ValueError(f"No JSON files found in {env_dir}")

        print(f"  Found {len(env_files)} cells")
        for f in env_files[:5]:
            print(f"    - {pathlib.Path(f).stem}")
        if len(env_files) > 5:
            print(f"    ... and {len(env_files) - 5} more")

        model, env, networks = make_multicell_model(
            env_files, reward_cfg, device_final, ppo_kwargs)
        output_name = "multi_cell"
    else:
        print(f"\n[Single-Cell Mode] Loading: {json_path}")
        graph_data = parse_transistor_json(json_path, verbose=True)
        graph_data["features"] = graph_data["features"].to(device_final)
        graph_data["adj"] = graph_data["adj"].to(device_final)

        model, env, networks = make_sb3_model(
            graph_data, reward_cfg, device_final, ppo_kwargs)
        output_name = json_path.stem

    encoder, policy_net, value_net = networks

    total_params = sum(p.numel() for p in encoder.parameters()) + \
        sum(p.numel() for p in policy_net.parameters()) + \
        sum(p.numel() for p in value_net.parameters())
    print(f"\n[Model] Total parameters: {total_params:,}")

    print(f"\n[Training] Starting {total_timesteps} timesteps...")

    best_csv = output_dir / f"{output_name}_best_placement.csv"

    callbacks = [
        TqdmCallback(total_timesteps),
        BestPlacementCallback(env, best_csv, verbose=1)
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False  # We use our own TQDM
    )

    model_path = output_dir / f"{output_name}_model.pth"
    torch.save({
        "encoder": encoder.state_dict(),
        "policy": policy_net.state_dict(),
        "value": value_net.state_dict(),
        "reward_cfg": reward_cfg
    }, model_path)

    print(f"\n[Complete]")
    print(f"  Model: {model_path}")
    print(f"  Best placement: {best_csv}")
    print(f"  TensorBoard: tensorboard --logdir {output_dir / 'tb'}")

    return model, env

###############################################################################
# 9. Main Entry Point
###############################################################################


def main():
    parser = argparse.ArgumentParser(
        description="Transistor Placement: GNN + Transformer + PPO (Multi-cell support)"
    )
    parser.add_argument("--input-file", type=pathlib.Path, default=None,
                        help="Single cell JSON file")
    parser.add_argument("--env-dir", type=pathlib.Path, default=None,
                        help="Directory with multiple cell JSONs (for multi-cell training)")
    parser.add_argument("--asap7-mode", action="store_true",
                        help="Parse from ASAP7 dataset format")
    parser.add_argument("--cell-name", type=str, default=None,
                        help="Cell name to extract (required for ASAP7 mode)")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("./output"),
                        help="Output directory")

    # Reward weights (Priority: Breaks > Dummy > Shared > HPWL)
    parser.add_argument("--w-break", type=float, default=100.0,
                        help="Weight for diffusion breaks (HIGHEST PRIORITY)")
    parser.add_argument("--w-dummy", type=float, default=50.0,
                        help="Weight for dummy gates (2nd priority)")
    parser.add_argument("--w-share", type=float, default=10.0,
                        help="Weight for shared diffusion (3rd priority)")
    parser.add_argument("--w-hpwl", type=float, default=2.0,
                        help="Weight for HPWL (4th priority)")
    parser.add_argument("--w-cdist", type=float, default=5.0,
                        help="Weight for column distance")

    # PPO hyperparameters
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--ent-coef", type=float, default=0.02,
                        help="Entropy coefficient (exploration)")

    # ASAP7 parameters
    parser.add_argument("--poly-pitch", type=float, default=0.054,
                        help="Poly pitch in um (ASAP7: 0.054)")
    parser.add_argument("--row-pitch", type=float, default=0.27,
                        help="Row pitch in um (ASAP7: 0.27)")

    parser.add_argument("--gpu", action="store_true",
                        help="Force GPU usage")

    args = parser.parse_args()

    # Validate arguments
    if args.input_file is None and args.env_dir is None:
        parser.error("Must provide either --input-file or --env-dir")

    if args.input_file and args.env_dir:
        parser.error("Cannot use both --input-file and --env-dir")

    use_device = device

    # Handle ASAP7 conversion
    input_json = None
    if args.asap7_mode:
        if args.cell_name is None:
            parser.error("--cell-name required when using --asap7-mode")
        if args.input_file is None:
            parser.error("--input-file required for ASAP7 mode")

        print(f"Converting ASAP7 cell '{args.cell_name}'...")
        env_json = parse_asap7_cell(args.input_file, args.cell_name,
                                    args.poly_pitch, args.row_pitch)

        converted_path = args.output_dir / f"{args.cell_name}_converted.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with open(converted_path, "w") as f:
            json.dump(env_json, f, indent=2)
        print(f"  Saved to: {converted_path}")
        input_json = converted_path
    elif args.input_file:
        input_json = args.input_file

    # Reward config
    reward_cfg = {
        "w_break": args.w_break,
        "w_dummy": args.w_dummy,
        "w_share": args.w_share,
        "w_hpwl": args.w_hpwl,
        "w_cdist": args.w_cdist,
    }

    # PPO config
    ppo_kwargs = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": args.ent_coef,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5
    }

    print("=" * 80)
    print("Transistor Placement: GNN + Transformer + PPO")
    print("Priority: Diffusion Breaks > Dummy > Shared > HPWL")
    print("=" * 80)

    # Device selection
    if args.gpu and torch.cuda.is_available():
        use_device = torch.device("cuda")
    else:
        use_device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

    # Train
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.env_dir is not None:
        # Multi-cell training (random cell per episode)
        model, env = train_transistor_placement(
            json_path=None,
            env_dir=args.env_dir,
            output_dir=output_dir,
            total_timesteps=args.timesteps,
            reward_cfg=reward_cfg,
            ppo_kwargs=ppo_kwargs,
            device_arg=use_device
        )
    else:
        # Single-cell training
        if input_json is None:
            raise ValueError("No input JSON resolved for single-cell mode.")
        model, env = train_transistor_placement(
            json_path=input_json,
            env_dir=None,
            output_dir=output_dir,
            total_timesteps=args.timesteps,
            reward_cfg=reward_cfg,
            ppo_kwargs=ppo_kwargs,
            device_arg=use_device
        )


if __name__ == "__main__":
    main()
