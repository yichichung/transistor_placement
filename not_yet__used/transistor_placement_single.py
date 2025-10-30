
"""
transistor_placement_single.py
Self-contained transistor-level (two-row, column-aligned) placement environment + training stub.
- Priority: minimize diffusion breaks
- Secondary: maximize shared diffusion length, reduce (non-essential) dummy
- Pairs (PMOS<->NMOS) are placed in the same column (mirrored)
- SB3 PPO training stub included (uses MlpPolicy by default; swap with your custom policy if needed)
"""

from __future__ import annotations
import json
import csv
from typing import Dict, Any, Tuple, List, Optional

# ------------------------------
# Parser (JSON -> graph-like dict)
# ------------------------------
import numpy as np

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
    deg = {i: 0 for i in range(len(devices))}
    for net in nets:
        dev_idxs = []
        for pin in net[:-1]:
            dev_name = pin.split(".")[0]
            if dev_name in name2idx:
                dev_idxs.append(name2idx[dev_name])
        dev_idxs = list(sorted(set(dev_idxs)))
        if len(dev_idxs) >= 2:
            net_pin_indices.append(dev_idxs)
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

    # Simple adjacency (row-normalized)
    N = len(devices)
    adj = np.eye(N, dtype=np.float32)
    for lst in net_pin_indices:
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                adj[a, b] = adj[b, a] = 1.0
    row_sums = adj.sum(axis=1, keepdims=True)
    adj = adj / (row_sums + 1e-8)

    edge_pairs = []
    for lst in net_pin_indices:
        for i in range(len(lst)):
            for j in range(i+1, len(lst)):
                a, b = lst[i], lst[j]
                edge_pairs.append((a, b)); edge_pairs.append((b, a))
    edge_index = np.array(edge_pairs, dtype=np.int64).T if edge_pairs else np.zeros((2,0), dtype=np.int64)

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

# ------------------------------
# Two-row placement Env (Gym/Gymnasium-compatible)
# ------------------------------
try:
    import gymnasium as gym
except ImportError:
    import gym

class TransistorPlacementEnv(gym.Env):
    """Two-row (PMOS/NMOS) column-aligned placement.
    Action: pick a device; if it has a pair, place both in same column (mirrored).
    Observation: (N,) 0/1 placed flags.
    Reward: episode-synced deltas of [breaks, shared, dummy, hpwl] with breaks prioritized.
    """
    metadata = {"render.modes": []}

    def __init__(self, graph: Dict[str, Any], reward_cfg: Dict[str, float]):
        super().__init__()
        self.graph = graph
        self.N = len(graph["devices"])
        self.devices = graph["devices"]
        self.pair_map = graph["pair_map"]
        self.grid = graph["grid"]

        self.col_idx = 0
        self.pmos_cols: List[Optional[int]] = []
        self.nmos_cols: List[Optional[int]] = []
        self.placed = np.zeros(self.N, dtype=np.int32)
        self.pos: Dict[int, Tuple[float, float, str, str, int]] = {i: None for i in range(self.N)}  # (x,y,orient,row,col)

        self.reward_cfg = reward_cfg
        self.prev_metrics_by_step: Dict[int, Dict[str, float]] = {}
        self.cur_step = 0

        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.N,), dtype=np.int32)
        self.action_space = gym.spaces.Discrete(self.N)

    # --- helpers ---
    def _pair_of(self, idx: int) -> Optional[int]:
        peer = self.pair_map.get(self.devices[idx]["name"])
        return None if peer is None else self.graph["name2idx"][peer]

    def _can_share(self, a: Optional[int], b: Optional[int], row_type: str) -> bool:
        if a is None or b is None: return False
        da, db = self.devices[a], self.devices[b]
        return da["type"] == row_type and db["type"] == row_type

    def _shared_amount(self, a: Optional[int], b: Optional[int]) -> float:
        if a is None or b is None: return 0.0
        da, db = self.devices[a], self.devices[b]
        return float(min(da.get("nf",1), db.get("nf",1)))

    def _count_breaks_and_shared(self, cols: List[Optional[int]], row_type: str) -> Tuple[int, float]:
        breaks, shared = 0, 0.0
        for i in range(len(cols)-1):
            L, R = cols[i], cols[i+1]
            if L is None or R is None:  # handled by dummy
                continue
            if self._can_share(L, R, row_type):
                shared += self._shared_amount(L, R)
            else:
                breaks += 1
        return breaks, shared

    def _estimate_dummy(self) -> Tuple[int, int]:
        dummy = 0
        for i in range(max(len(self.pmos_cols), len(self.nmos_cols))):
            p = self.pmos_cols[i] if i < len(self.pmos_cols) else None
            n = self.nmos_cols[i] if i < len(self.nmos_cols) else None
            if (p is None) ^ (n is None):
                dummy += 1
        eff_dummy = 0  # treat alignment dummies as useful initially
        return dummy, eff_dummy

    def _compute_hpwl(self) -> float:
        xy = {i: (p[0], p[1]) for i, p in self.pos.items() if p is not None}
        total = 0.0
        for dev_idxs in self.graph["nets_index"]:
            xs, ys = [], []
            for i in dev_idxs:
                if i in xy:
                    x,y = xy[i]; xs.append(x); ys.append(y)
            if len(xs) >= 2:
                total += (max(xs)-min(xs)) + (max(ys)-min(ys))
        return total

    def _metrics(self) -> Dict[str, float]:
        bP, sP = self._count_breaks_and_shared(self.pmos_cols, "PMOS")
        bN, sN = self._count_breaks_and_shared(self.nmos_cols, "NMOS")
        dummy, eff_dummy = self._estimate_dummy()
        return {
            "breaks": bP + bN,
            "shared": sP + sN,
            "dummy": float(dummy),
            "eff_dummy": float(eff_dummy),
            "hpwl": self._compute_hpwl()
        }

    # --- gym api ---
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.col_idx = 0
        self.pmos_cols.clear(); self.nmos_cols.clear()
        self.placed[:] = 0
        self.pos = {i: None for i in range(self.N)}
        self.prev_metrics_by_step.clear()
        self.cur_step = 0
        return self.placed.copy(), {}

    def get_action_mask(self) -> np.ndarray:
        mask = np.ones(self.N, dtype=np.int8)
        mask[self.placed == 1] = 0
        return mask

    def step(self, action: int):
        done = False; info = {}

        if self.placed[action] == 1:
            return self.placed.copy(), -1.0, done, False, {"invalid_action": True, "mask": self.get_action_mask()}

        def place_to_col(idx: int, row_type: str, col: int):
            x = self.grid["poly_pitch"] * col
            y = self.grid["y_pmos"] if row_type == "PMOS" else self.grid["y_nmos"]
            orient = "MX" if row_type == "PMOS" else "R0"
            self.pos[idx] = (x, y, orient, row_type, col)
            self.placed[idx] = 1

        # ensure this column exists
        if self.col_idx == len(self.pmos_cols):
            self.pmos_cols.append(None); self.nmos_cols.append(None)

        dev = self.devices[action]
        row = dev["type"]
        if row == "PMOS":
            self.pmos_cols[self.col_idx] = action
            place_to_col(action, "PMOS", self.col_idx)
        else:
            self.nmos_cols[self.col_idx] = action
            place_to_col(action, "NMOS", self.col_idx)

        # mirror pair into same column if present
        peer = self._pair_of(action)
        if peer is not None and self.placed[peer] == 0:
            peer_row = self.devices[peer]["type"]
            if peer_row == "PMOS":
                self.pmos_cols[self.col_idx] = peer
                place_to_col(peer, "PMOS", self.col_idx)
            else:
                self.nmos_cols[self.col_idx] = peer
                place_to_col(peer, "NMOS", self.col_idx)

        self.col_idx += 1
        self.cur_step += 1

        cur = self._metrics()
        prev = self.prev_metrics_by_step.get(self.cur_step, None)
        if prev is None:
            delta = {k: 0.0 for k in cur.keys()}
        else:
            delta = {k: cur[k] - prev[k]}
        self.prev_metrics_by_step[self.cur_step] = cur

        w_break = float(self.reward_cfg.get("w_break", 10.0))
        w_share = float(self.reward_cfg.get("w_share", 4.0))
        w_dummy = float(self.reward_cfg.get("w_dummy_eff", 0.0))
        w_hpwl = float(self.reward_cfg.get("w_hpwl", 0.5))

        reward = (
            - w_break * delta["breaks"]
            + w_share * delta["shared"]
            - w_dummy * delta["eff_dummy"]
            + w_hpwl * (-delta["hpwl"])
        )

        done = bool(np.all(self.placed == 1))
        info["mask"] = self.get_action_mask()
        info["metrics"] = cur
        return self.placed.copy(), float(reward), done, False, info

# ------------------------------
# CSV export
# ------------------------------
def export_transistor_csv(path: str, env: TransistorPlacementEnv) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["device_name","device_type","row","column","x","y","orient","w","l","nf","shared_with","is_dummy"])
        for i, d in enumerate(env.devices):
            p = env.pos.get(i)
            if p is None: continue
            x, y, orient, row, col = p
            w.writerow([
                d["name"], d["type"], row, col, x, y, orient,
                d.get("w",1.0), d.get("l",0.05), d.get("nf",1),
                "", 0
            ])

# ------------------------------
# SB3 training stub
# ------------------------------
def train_with_sb3(json_path: str, total_timesteps: int = 50_000, policy="MlpPolicy"):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    graph = parse_transistor_json(json_path)
    reward_cfg = dict(w_break=10.0, w_share=4.0, w_dummy_eff=0.0, w_hpwl=0.5)

    def _make():
        return TransistorPlacementEnv(graph, reward_cfg)

    env = DummyVecEnv([_make])
    model = PPO(policy=policy, env=env, n_steps=2048, batch_size=64, learning_rate=3e-4)
    model.learn(total_timesteps=total_timesteps)
    return model

if __name__ == "__main__":
    import argparse, os
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="sample_circuit.json")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--export_csv", type=str, default="placement.csv")
    args = parser.parse_args()

    # If sample json doesn't exist, create a small demo
    if not os.path.exists(args.json):
        sample = {
          "devices": [
            {"name":"M1","type":"NMOS","w":1.0,"l":0.05,"nf":2,"vt":"SVT"},
            {"name":"M2","type":"PMOS","w":1.2,"l":0.05,"nf":2,"vt":"SVT"},
            {"name":"M3","type":"NMOS","w":0.8,"l":0.05,"nf":1,"vt":"SVT"},
            {"name":"M4","type":"PMOS","w":0.8,"l":0.05,"nf":1,"vt":"SVT"}
          ],
          "nets": [
            ["M1.D","M2.S","out"], ["M1.G","M2.G","in"], ["M1.S","vss"], ["M2.D","vdd"],
            ["M3.D","M4.S","out2"], ["M3.G","M4.G","in2"], ["M3.S","vss"], ["M4.D","vdd"]
          ],
          "constraints": {
            "pair_map": {"M1":"M2","M2":"M1","M3":"M4","M4":"M3"},
            "row_pitch": 1.0, "poly_pitch": 0.1, "must_align_gates": True
          }
        }
        with open(args.json, "w") as f:
            json.dump(sample, f, indent=2)

    # Train (uses simple MlpPolicy for demo — swap with your custom Transformer policy)
    model = train_with_sb3(args.json, total_timesteps=args.timesteps)

    # NOTE: To export CSV you need access to the underlying env instance.
    # Below is a quick rollout to populate positions and then export:
    from stable_baselines3.common.vec_env import DummyVecEnv
    graph = parse_transistor_json(args.json)
    reward_cfg = dict(w_break=10.0, w_share=4.0, w_dummy_eff=0.0, w_hpwl=0.5)
    def _make():
        return TransistorPlacementEnv(graph, reward_cfg)
    eval_env = DummyVecEnv([_make])
    obs, _ = eval_env.reset()
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, trunc, info = eval_env.step(action)
        if done[0]:
            break
    # Grab the unwrapped env to export CSV
    env0 = eval_env.envs[0]
    export_transistor_csv(args.export_csv, env0)
    print(f"Exported CSV to {args.export_csv}")
