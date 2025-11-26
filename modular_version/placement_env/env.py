"""
Reinforcement Learning Environment for Transistor Placement.
Defines state, actions, rewards, and constraints.
"""
from __future__ import annotations
import random
import pathlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
except ImportError:
    import gym as gym

from .data_parser import TransistorNode, parse_transistor_json


class TransistorPlacementEnv(gym.Env):
    """
    Core RL environment for transistor placement.

    State: Placement sequence (which devices have been placed)
    Action: Select next device to place
    Reward: Based on HPWL, breaks, shared diffusions, dummies, and pair distance
    """
    metadata = {"render.modes": []}

    def __init__(self, graph_data: Dict, reward_cfg: Dict, device=None):
        super().__init__()
        self._rebuild_from_dict(graph_data)
        self.device_tensor = device
        self.reward_cfg = reward_cfg

        self.observation_space = gym.spaces.Box(
            low=-1, high=self.N_max, shape=(self.N_max,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(self.N_max)

    def _rebuild_from_dict(self, graph_data: Dict):
        """Rebuild environment from graph data dictionary."""
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
        """Validate and normalize NMOS-PMOS pairing constraints."""
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
        """Get the paired device index (NMOS-PMOS pair)."""
        if idx >= len(self.devices):
            return None
        name = self.devices[idx]["name"]
        peer_name = self.pair_map.get(name)
        if peer_name is None:
            return None
        return self.graph["name2idx"].get(peer_name)

    def _can_share(self, a_idx: Optional[int], b_idx: Optional[int], row_type: str) -> bool:
        """Check if two adjacent devices can share diffusion."""
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
        """Calculate the amount of shared diffusion (in finger units)."""
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
        """Count diffusion breaks and shared diffusion in a row."""
        breaks, shared = 0, 0.0
        for i in range(len(cols) - 1):
            if self._can_share(cols[i], cols[i+1], row_type):
                shared += self._shared_amount(cols[i], cols[i+1])
            else:
                if cols[i] is not None and cols[i+1] is not None:
                    breaks += 1
        return breaks, shared

    def _estimate_dummy(self) -> Tuple[int, int]:
        """Estimate number of dummy transistors needed for alignment."""
        dummy = 0
        max_cols = max(len(self.pmos_cols), len(self.nmos_cols))
        for i in range(max_cols):
            p = self.pmos_cols[i] if i < len(self.pmos_cols) else None
            n = self.nmos_cols[i] if i < len(self.nmos_cols) else None
            if (p is None) ^ (n is None):
                dummy += 1
        return dummy, 0

    def _compute_hpwl(self) -> float:
        """Compute normalized Half-Perimeter Wire Length."""
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
        """Calculate average column distance between paired devices."""
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
                if p_idx in self.positions and n_idx in self.positions:
                    p_col = self.positions[p_idx][4]
                    n_col = self.positions[n_idx][4]
                    if isinstance(p_col, int) and isinstance(n_col, int):
                        total += abs(p_col - n_col)
                        count += 1
        return float(total) / max(count, 1)

    def _metrics(self) -> Dict:
        """Compute all placement metrics."""
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
        """Get valid action mask (1 = valid, 0 = invalid)."""
        mask = np.zeros(self.N_max, dtype=np.float32)
        mask[:self.N] = 1.0
        mask[:self.N][self.placed == 1] = 0.0
        return mask

    def get_current_placement_dicts(self) -> List[Dict]:
        """Export current placement as list of device dictionaries."""
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
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.col_idx = 0
        self.pmos_cols, self.nmos_cols = [], []
        self.placed = np.zeros(self.N, dtype=np.int32)
        self.positions = {}
        self.sequence = []
        self.episode_steps = 0
        self._last_metrics = None

        obs = np.zeros(self.N_max, dtype=np.float32)
        info = {"mask": self.get_action_mask()}
        return obs, info

    def step(self, action: int):
        """
        Execute one placement action.

        Args:
            action: Device index to place

        Returns:
            obs: Updated observation
            reward: Step reward
            terminated: Episode done flag
            truncated: Early termination flag
            info: Additional information
        """
        done = False
        info = {}

        # Invalid action check
        if not (0 <= action < self.N) or self.placed[action] == 1:
            obs = np.zeros(self.N_max, dtype=np.float32)
            for i, idx in enumerate(self.sequence):
                if idx < self.N_max:
                    obs[idx] = float(i + 1)
            info["invalid_action"] = True
            info["mask"] = self.get_action_mask()
            return obs, -1.0, False, False, info

        def _place_to_col(idx: int, row_type: str, col: int):
            """Helper to place a device at specific column."""
            x = self.grid["poly_pitch"] * col
            y = self.grid["y_pmos"] if row_type == "PMOS" else self.grid["y_nmos"]
            orient = "MX" if row_type == "PMOS" else "R0"
            self.positions[idx] = (x, y, orient, row_type, col)
            self.placed[idx] = 1
            self.sequence.append(idx)

        # Extend column lists if needed
        if self.col_idx >= len(self.pmos_cols):
            self.pmos_cols.append(None)
            self.nmos_cols.append(None)

        # Place selected device
        dev = self.devices[action]
        row = dev["type"]
        if row == "PMOS":
            self.pmos_cols[self.col_idx] = action
            _place_to_col(action, "PMOS", self.col_idx)
        else:
            self.nmos_cols[self.col_idx] = action
            _place_to_col(action, "NMOS", self.col_idx)

        # Auto-place paired device if exists
        peer_name = self.pair_map.get(dev["name"])
        if peer_name is not None:
            peer_idx = self.graph["name2idx"].get(peer_name)
            if peer_idx is not None and self.placed[peer_idx] == 0:
                peer_row = self.devices[peer_idx]["type"]
                if peer_row == "PMOS" and self.pmos_cols[self.col_idx] is None:
                    self.pmos_cols[self.col_idx] = peer_idx
                    _place_to_col(peer_idx, "PMOS", self.col_idx)
                elif peer_row == "NMOS" and self.nmos_cols[self.col_idx] is None:
                    self.nmos_cols[self.col_idx] = peer_idx
                    _place_to_col(peer_idx, "NMOS", self.col_idx)

        self.col_idx += 1
        self.episode_steps += 1

        # Compute reward
        cur = self._metrics()
        if self._last_metrics is None:
            delta = {k: 0.0 for k in cur.keys()}
        else:
            delta = {k: cur[k] - self._last_metrics[k] for k in cur.keys()}
        self._last_metrics = cur

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

        # Check if episode complete
        done = bool(np.all(self.placed == 1))
        if done:
            final = cur
            reward += (
                -w_break * final["breaks"] +
                -w_dummy * final["dummy"] +
                w_share * final["shared"] -
                w_hpwl * final["hpwl"] -
                w_cdist * final["col_dist"]
            )
            info["final_metrics"] = final
            info["episode_steps"] = self.episode_steps

            # Store placement data for callbacks
            self.current_placement = self.get_current_placement_dicts()
            info["placement"] = self.current_placement
            info["cell_name"] = self.graph.get("cell_name", "unknown_cell")

            print(f"[Done] Steps={self.episode_steps} | Breaks={final['breaks']} | "
                  f"Dummy={final['dummy']} | Shared={final['shared']:.1f} | "
                  f"HPWL={final['hpwl']:.3f} | ColDist={final['col_dist']:.2f}")

        # Build observation
        obs = np.zeros(self.N_max, dtype=np.float32)
        for i, idx in enumerate(self.sequence):
            if idx < self.N_max:
                obs[idx] = float(i + 1)
        info["mask"] = self.get_action_mask()
        return obs, float(reward), bool(done), False, info


class RandomMultiCellEnv(TransistorPlacementEnv):
    """
    Multi-cell wrapper that randomly samples from multiple circuits.
    Used for training on diverse cell libraries.
    """
    def __init__(self, env_files: List[Union[str, pathlib.Path]], reward_cfg: Dict, device=None):
        self.env_files = [pathlib.Path(p) for p in env_files]
        self.reward_cfg = reward_cfg
        self.device_tensor = device

        self._graphs = {}
        Ns = []
        for p in self.env_files:
            g = parse_transistor_json(p, verbose=False)
            self._graphs[str(p)] = g
            Ns.append(len(g["devices"]))
        self.N_max = int(max(Ns))

        first_data = self._graphs[str(self.env_files[0])]
        super().__init__(first_data, reward_cfg, device)
        self.N_max = int(self.N_max)

        self.observation_space = gym.spaces.Box(
            low=-1, high=self.N_max, shape=(self.N_max,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(self.N_max)

        print(
            f"[Multi-Cell] Loaded {len(self.env_files)} cells (N_max={self.N_max})")
        for f in self.env_files[:5]:
            print(f"  - {f.stem}")
        if len(self.env_files) > 5:
            print(f"  ... and {len(self.env_files) - 5} more")

    def reset(self, *, seed=None, options=None):
        """Reset and randomly select a new cell."""
        selected_file = random.choice(self.env_files)
        graph_data = self._graphs.get(str(selected_file))
        if graph_data is None:
            graph_data = parse_transistor_json(selected_file, verbose=False)
            self._graphs[str(selected_file)] = graph_data

        self._rebuild_from_dict(graph_data)
        return super().reset(seed=seed, options=options)
