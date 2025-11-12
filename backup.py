"""
Transistor Placement using GNN + Transformer + PPO (SB3)
COMPLETE FIX:
- Intelligent gate-based pairing
- Proper diffusion break detection with S/D mirroring
- Fixed N_max dimension handling
- TensorBoard + Multi-cell training
"""
from __future__ import annotations
import json
import argparse
import pathlib
import csv
import math
import random
from typing import Optional, Tuple, Dict, List, Any, Union
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
MOS_TYPES = ["NMOS", "PMOS"]

###############################################################################
# 1. Intelligent Pairing + ASAP7 Parser
###############################################################################


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


def _build_intelligent_pair_map(
    rowN: List[dict],
    rowP: List[dict],
    name_map_N: Dict[int, str],
    name_map_P: Dict[int, str]
) -> Dict[str, str]:
    """
    INTELLIGENT PAIRING: Match NMOS-PMOS by gate_net (REQUIRED)
    Tie-breaking: drain/source connectivity + width similarity
    """
    from math import inf

    # Index PMOS by gate_net
    p_by_gate = defaultdict(list)
    for j, tp in enumerate(rowP):
        g = tp.get("gate_net")
        if g is not None:
            p_by_gate[g].append(j)

    used_p = set()
    pair_map: Dict[str, str] = {}

    # Round 1: Unique gate match
    for i, tn in enumerate(rowN):
        g = tn.get("gate_net")
        cand = [j for j in p_by_gate.get(g, []) if j not in used_p]

        if len(cand) == 1:
            j = cand[0]
            pair_map[name_map_N[i]] = name_map_P[j]
            pair_map[name_map_P[j]] = name_map_N[i]
            used_p.add(j)

    # Round 2: Multi-candidate scoring
    def score_pair(tn: dict, tp: dict) -> Tuple[int, float]:
        score = 0
        dn, sn = tn.get("net_drn"), tn.get("net_src")
        dp, sp = tp.get("net_drn"), tp.get("net_src")

        if dn is not None and (dn == dp or dn == sp):
            score += 2
        if sn is not None and (sn == dp or sn == sp):
            score += 1

        wdiff = abs(float(tp.get("W_norm", 1.0)) - float(tn.get("W_norm", 1.0)))
        return (score, -wdiff)

    for i, tn in enumerate(rowN):
        nname = name_map_N[i]
        if nname in pair_map:
            continue

        g = tn.get("gate_net")
        cand = [j for j in p_by_gate.get(g, []) if j not in used_p]

        if cand:
            best = (-1, -inf, None)
            for j in cand:
                sc = score_pair(tn, rowP[j])
                if sc > best[:2]:
                    best = (sc[0], sc[1], j)

            j = best[2]
            if j is not None:
                pair_map[nname] = name_map_P[j]
                pair_map[name_map_P[j]] = nname
                used_p.add(j)

    # Round 3: Remaining sequential pairing
    remaining_p = [j for j in range(len(rowP)) if j not in used_p]
    idx = 0
    for i in range(len(rowN)):
        nname = name_map_N[i]
        if nname not in pair_map and idx < len(remaining_p):
            j = remaining_p[idx]
            idx += 1
            pair_map[nname] = name_map_P[j]
            pair_map[name_map_P[j]] = nname

    return pair_map


def parse_asap7_cell(asap7_json: pathlib.Path, cell_name: str,
                     poly_pitch: float = 0.056, row_pitch: float = 1.0) -> dict:
    """Parse ASAP7 dataset with intelligent pairing"""
    with open(asap7_json, "r") as f:
        data = json.load(f)

    if cell_name not in data:
        available = list(data.keys())[:10]
        raise ValueError(f"Cell '{cell_name}' not found. Available: {available}")

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
            "name": name, "type": "NMOS",
            "w": float(t.get("W_norm", 1e-7)), "l": 0.05,
            "nf": 1, "vt": "SVT"
        })

    for i, t in enumerate(row_P):
        name = f"MP{i}"
        name_map_P[i] = name
        devices.append({
            "name": name, "type": "PMOS",
            "w": float(t.get("W_norm", 1e-7)), "l": 0.05,
            "nf": 1, "vt": "SVT"
        })

    # Build nets
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

    pair_map = _build_intelligent_pair_map(row_N, row_P, name_map_N, name_map_P)

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


def build_graph_from_nets(devices: List[dict], nets: List[List[str]]) -> torch.Tensor:
    """Build adjacency matrix using clique expansion"""
    name_to_idx = {d["name"]: i for i, d in enumerate(devices)}
    N = len(devices)
    adj_matrix = np.zeros((N, N), dtype=np.float32)

    def pin_to_dev(pin: str) -> Optional[int]:
        if not isinstance(pin, str) or "." not in pin:
            return None
        dev_name = pin.split(".")[0]
        return name_to_idx.get(dev_name, None)

    for net in nets:
        pins = [p for p in net[:-1] if isinstance(p, str) and "." in p]
        if not pins:
            continue

        dev_indices = []
        for pin in pins:
            idx = pin_to_dev(pin)
            if idx is not None:
                dev_indices.append(idx)

        dev_indices = sorted(set(dev_indices))
        if len(dev_indices) < 2:
            continue

        # Clique expansion
        for i in range(len(dev_indices)):
            for j in range(i + 1, len(dev_indices)):
                u, v = dev_indices[i], dev_indices[j]
                if u != v:
                    adj_matrix[u, v] = 1.0
                    adj_matrix[v, u] = 1.0

    # Add self-loops
    for i in range(N):
        adj_matrix[i, i] = 1.0

    # Row normalization
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    adj_matrix = adj_matrix / (row_sums + 1e-8)

    return torch.tensor(adj_matrix, dtype=torch.float32)


def parse_transistor_json(path: Union[str, pathlib.Path], verbose: bool = False) -> dict:
    """Parse environment JSON with pin-to-net mapping"""
    path = pathlib.Path(path)

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

    adj = build_graph_from_nets(devices, nets)

    # Build pin-to-net mapping
    pin_nets = {i: {"S": None, "D": None, "G": None} for i in range(len(devices))}

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
                parts = token.split(".", 1)
                if len(parts) == 2:
                    dev, pin = parts
                    set_pin(dev, pin, net_name)

    # Build netlist
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

    # Node features
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
        print(f"[Graph] Nodes={N}, Nets={len(net_pin_indices)}, Edges={num_edges}")

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
            freqs = col_indices.new_tensor([2.0**k for k in range(self.num_bands)]) * (2 * math.pi)
            col_w = col_indices * freqs
            feats = torch.cat([col_indices, torch.sin(col_w), torch.cos(col_w)], dim=-1)
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
        attn_out, _ = self.mha(q, kv, kv, attn_mask=attn_mask, need_weights=False)
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
        self.gamma = nn.Parameter(torch.tensor(float(gamma)), requires_grad=False)
        self.sigma = nn.Parameter(torch.tensor(float(sigma)), requires_grad=False)

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
                hist_cols = torch.tensor(hist_cols_list, device=device, dtype=dtype)
                pe1d = self.pos1d(hist_cols)
                context = context.squeeze(0) + pe1d
                context = F.layer_norm(context, (self.embed_dim,)).unsqueeze(0)

                attn_mask = self._build_bias_1d(next_cols, hist_cols, N, device, dtype)
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
# 5. Environment
###############################################################################


class TransistorPlacementEnv(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, graph_data: Dict, reward_cfg: Dict, device=None):
        super().__init__()
        self._rebuild_from_dict(graph_data)
        self.device_tensor = device or torch.device("cpu")
        self.reward_cfg = reward_cfg

        self.observation_space = gym.spaces.Box(
            low=-1, high=self.N_max, shape=(self.N_max,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.N_max)

    def _rebuild_from_dict(self, graph_data: Dict):
        self.graph = graph_data
        self.nodes: List[TransistorNode] = graph_data["nodes"]
        self.N = len(self.nodes)

        if not hasattr(self, "N_max"):
            self.N_max = self.N

        self.devices = graph_data["devices"]
        self.pair_map = graph_data["pair_map"]
        self.grid = graph_data["grid"]
        self.netlist = graph_data["netlist"]
        self.pin_nets = graph_data.get("pin_nets", {})

        self._normalize_pair_map()

        self.col_idx = 0
        self.pmos_cols = []
        self.nmos_cols = []
        self.placed = np.zeros(self.N, dtype=np.int32)
        self.positions = {}
        self.sequence = []
        self.episode_steps = 0
        self._last_metrics = None
        self.current_placement = []

    def _normalize_pair_map(self):
        name2idx = self.graph["name2idx"]
        clean = {}
        seen = set()

        for a, b in self.pair_map.items():
            if a not in name2idx or b not in name2idx:
                continue
            ia, ib = name2idx[a], name2idx[b]
            if ia >= len(self.devices) or ib >= len(self.devices):
                continue
            ta, tb = self.devices[ia]["type"], self.devices[ib]["type"]

            if {ta, tb} != {"NMOS", "PMOS"}:
                continue

            key = tuple(sorted([a, b]))
            if key in seen:
                continue
            seen.add(key)

            clean[a] = b
            clean[b] = a

        self.pair_map = clean

    def _pair_of(self, idx: int) -> Optional[int]:
        if idx >= len(self.devices):
            return None
        name = self.devices[idx]["name"]
        peer_name = self.pair_map.get(name)
        if peer_name is None:
            return None
        return self.graph["name2idx"].get(peer_name)

    def _can_share(self, a_idx: Optional[int], b_idx: Optional[int], row_type: str) -> bool:
        if a_idx is None or b_idx is None:
            return False
        if a_idx >= len(self.devices) or b_idx >= len(self.devices):
            return False

        a = self.devices[a_idx]
        b = self.devices[b_idx]

        if a["type"] != row_type or b["type"] != row_type:
            return False

        pa = self.pin_nets.get(a_idx, {})
        pb = self.pin_nets.get(b_idx, {})
        aS, aD = pa.get("S"), pa.get("D")
        bS, bD = pb.get("S"), pb.get("D")

        for x in (aS, aD):
            for y in (bS, bD):
                if x and y and x == y:
                    return True
        return False

    def _shared_amount(self, a_idx: Optional[int], b_idx: Optional[int]) -> float:
        if a_idx is None or b_idx is None:
            return 0.0
        if a_idx >= len(self.devices) or b_idx >= len(self.devices):
            return 0.0

        row_type = self.devices[a_idx]["type"]
        if not self._can_share(a_idx, b_idx, row_type):
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
        cell_w = self.grid["poly_pitch"] * max(len(self.pmos_cols), len(self.nmos_cols))
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
                total += ((max(xs) - min(xs)) + (max(ys) - min(ys))) / norm_factor
        return total

    def _col_distance_cost(self) -> float:
        name2idx = self.graph["name2idx"]
        total = 0.0
        count = 0

        processed = set()
        for p_name, n_name in self.pair_map.items():
            pair_key = tuple(sorted([p_name, n_name]))
            if pair_key in processed:
                continue
            processed.add(pair_key)

            if p_name in name2idx and n_name in name2idx:
                p_idx = name2idx[p_name]
                n_idx = name2idx[n_name]