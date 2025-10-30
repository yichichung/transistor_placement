
# env_transistor.py
# Two-row (PMOS/NMOS) column-aligned transistor-placement Gym environment focused on
# minimizing diffusion breaks first, maximizing shared diffusion, and reducing dummy gates.

import numpy as np
from typing import Dict, Any, Tuple, List, Optional

try:
    import gymnasium as gym  # Gymnasium preferred
except ImportError:
    import gym as gym

class TransistorPlacementEnv(gym.Env):
    """
    Single PMOS row + single NMOS row; columns advance left->right.
    Action: pick a device index to place; if the chosen device has a pair (PMOS<->NMOS),
    the environment places the pair in the same column (mirrored) automatically.
    Observation: (N,) vector of 0/1 (unplaced/placed).

    Reward: Episode-synced step-wise differences on metrics (breaks/shared/dummy/hpwl),
    prioritizing diffusion-break minimization.
    """

    metadata = {"render.modes": []}

    def __init__(self, graph: Dict[str, Any], reward_cfg: Dict[str, float]):
        super().__init__()
        self.graph = graph
        self.N = len(graph["devices"])
        self.devices = graph["devices"]
        self.pair_map = graph["pair_map"]
        self.grid = graph["grid"]

        # Layout state
        self.col_idx = 0
        self.pmos_cols: List[Optional[int]] = []
        self.nmos_cols: List[Optional[int]] = []
        self.placed = np.zeros(self.N, dtype=np.int32)  # 0/1
        # (x, y, orient, row, col)
        self.pos: Dict[int, Tuple[float, float, str, str, int]] = {i: None for i in range(self.N)}  # type: ignore

        # Episode/Reward bookkeeping
        self.reward_cfg = reward_cfg
        self.prev_metrics_by_step: Dict[int, Dict[str, float]] = {}
        self.cur_step = 0

        # Gym spaces
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.N,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(self.N)

    # ---------- geometry & metric helpers ----------
    def _pair_of(self, idx: int) -> Optional[int]:
        name = self.devices[idx]["name"]
        peer = self.pair_map.get(name)
        if peer is None:
            return None
        return self.graph["name2idx"][peer]

    def _can_share(self, a_idx: Optional[int], b_idx: Optional[int], row_type: str) -> bool:
        if a_idx is None or b_idx is None:
            return False
        a = self.devices[a_idx]; b = self.devices[b_idx]
        if a["type"] != row_type or b["type"] != row_type:
            return False
        # Minimal version: same polarity on same row ⇒ shareable
        return True

    def _shared_amount(self, a_idx: Optional[int], b_idx: Optional[int]) -> float:
        if a_idx is None or b_idx is None:
            return 0.0
        a = self.devices[a_idx]; b = self.devices[b_idx]
        return float(min(a.get("nf", 1), b.get("nf", 1)))

    def _count_breaks_and_shared(self, cols: List[Optional[int]], row_type: str) -> Tuple[int, float]:
        breaks, shared = 0, 0.0
        for i in range(len(cols) - 1):
            left, right = cols[i], cols[i+1]
            if left is None or right is None:
                # empty column on at least one side: not a break, handled by dummy
                continue
            if self._can_share(left, right, row_type):
                shared += self._shared_amount(left, right)
            else:
                breaks += 1
        return breaks, shared

    def _estimate_dummy(self) -> Tuple[int, int]:
        # Column-wise alignment: if only one row has a device ⇒ one dummy needed on the other row
        dummy = 0
        for i in range(max(len(self.pmos_cols), len(self.nmos_cols))):
            p = self.pmos_cols[i] if i < len(self.pmos_cols) else None
            n = self.nmos_cols[i] if i < len(self.nmos_cols) else None
            if (p is None) ^ (n is None):
                dummy += 1
        eff_dummy = 0  # simplest version: treat all as "useful"; tune later
        return dummy, eff_dummy

    def _compute_hpwl(self) -> float:
        # Use device centers (x,y) as pins for rough HPWL estimate
        xy = {i: (p[0], p[1]) for i, p in self.pos.items() if p is not None}
        total = 0.0
        for dev_idxs in self.graph["nets_index"]:
            xs, ys = [], []
            for i in dev_idxs:
                if i in xy:
                    x, y = xy[i]; xs.append(x); ys.append(y)
            if len(xs) >= 2:
                total += (max(xs) - min(xs)) + (max(ys) - min(ys))
        return total

    def _metrics(self) -> Dict[str, float]:
        b_p, s_p = self._count_breaks_and_shared(self.pmos_cols, "PMOS")
        b_n, s_n = self._count_breaks_and_shared(self.nmos_cols, "NMOS")
        dummy, eff_dummy = self._estimate_dummy()
        hpwl = self._compute_hpwl()
        return {
            "breaks": b_p + b_n,
            "shared": s_p + s_n,
            "dummy": float(dummy),
            "eff_dummy": float(eff_dummy),
            "hpwl": hpwl,
        }

    # ---------- gym api ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.col_idx = 0
        self.pmos_cols.clear(); self.nmos_cols.clear()
        self.placed[:] = 0
        self.pos = {i: None for i in range(self.N)}
        self.prev_metrics_by_step.clear()
        self.cur_step = 0
        return self._obs(), {}

    def _obs(self):
        return self.placed.copy()

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(self.N, dtype=np.int8)
        mask[self.placed == 1] = 0
        return mask

    def step(self, action: int):
        done = False; info = {}

        if self.placed[action] == 1:
            # invalid: small negative
            return self._obs(), -1.0, done, False, {"invalid_action": True, "mask": self.get_action_mask()}

        def place_to_col(idx: int, row_type: str, col: int):
            x = self.grid["poly_pitch"] * col
            y = self.grid["y_pmos"] if row_type == "PMOS" else self.grid["y_nmos"]
            orient = "MX" if row_type == "PMOS" else "R0"
            self.pos[idx] = (x, y, orient, row_type, col)
            self.placed[idx] = 1

        # Ensure this column exists
        if self.col_idx == len(self.pmos_cols):
            self.pmos_cols.append(None)
            self.nmos_cols.append(None)

        dev = self.devices[action]
        row = dev["type"]
        if row == "PMOS":
            self.pmos_cols[self.col_idx] = action
            place_to_col(action, "PMOS", self.col_idx)
        else:
            self.nmos_cols[self.col_idx] = action
            place_to_col(action, "NMOS", self.col_idx)

        # Mirror pair (same column) if not yet placed
        peer = self._pair_of(action)
        if peer is not None and self.placed[peer] == 0:
            peer_row = self.devices[peer]["type"]
            if peer_row == "PMOS":
                self.pmos_cols[self.col_idx] = peer
                place_to_col(peer, "PMOS", self.col_idx)
            else:
                self.nmos_cols[self.col_idx] = peer
                place_to_col(peer, "NMOS", self.col_idx)

        # Advance to next column
        self.col_idx += 1
        self.cur_step += 1

        # Compute reward as episode-synced delta
        cur = self._metrics()
        prev = self.prev_metrics_by_step.get(self.cur_step, None)
        if prev is None:
            delta = {k: 0.0 for k in cur.keys()}
        else:
            delta = {k: cur[k] - prev[k] for k in cur.keys()}
        self.prev_metrics_by_step[self.cur_step] = cur

        w_break  = float(self.reward_cfg.get("w_break", 10.0))
        w_share  = float(self.reward_cfg.get("w_share", 4.0))
        w_dummy  = float(self.reward_cfg.get("w_dummy_eff", 0.0))
        w_hpwl   = float(self.reward_cfg.get("w_hpwl", 0.5))

        reward = (
            - w_break * delta["breaks"]
            + w_share * delta["shared"]
            - w_dummy * delta["eff_dummy"]
            + w_hpwl * (-delta["hpwl"])
        )

        done = bool(np.all(self.placed == 1))
        info["mask"] = self.get_action_mask()
        info["metrics"] = cur
        return self._obs(), float(reward), done, False, info
