"""
Transistor Placement V2.1: Final Complete + Auto-Save Best Solutions
--------------------------------------------------------------------------
Features:
1. Dual-Rail & Flip: Supports PMOS/NMOS row separation and device orientation.
2. Physical Awareness: Correct NF width logic and physical pin matching.
3. Auto-Save: Tracks best solution per cell (min width, min HPWL) and saves to CSV.
4. JIT Behavior Cloning: Warm start from expert data.
5. Robustness: All previous bug fixes (masks, types, graph weights) included.
"""
from __future__ import annotations
import json
import argparse
import pathlib
import os
import math
import random
import csv
from typing import Optional, Tuple, Dict, List, Any, Union
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

import gymnasium as gym
from gymnasium import spaces

try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
    from sb3_contrib.common.wrappers import ActionMasker
except ImportError:
    raise ImportError("Please run: pip install sb3-contrib")

from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CallbackList

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GNN_output_dim = 128

###############################################################################
# 1. Data Structures & Graph
###############################################################################


def _want_pin_token(tok: str) -> bool:
    if not isinstance(tok, str) or "." not in tok:
        return False
    _, pin = tok.split(".", 1)
    return pin.upper() in {"S", "D", "G"}


def _is_power_net(net_name: str) -> bool:
    return net_name in {"VDD", "VSS", "vdd", "vss", "VDD!", "VSS!"}


def build_graph_from_nets(devices: List[dict], nets: List[List[str]]) -> torch.Tensor:
    """
    Builds adjacency matrix.
    Weight 1.0 for S/D connections (strong).
    Weight 0.5 for Gate connections (weak).
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

        pin_nodes = []
        valid_tokens = [tok for tok in net[:-1] if _want_pin_token(tok)]
        for tok in valid_tokens:
            dev_name, pin_type = tok.split(".", 1)
            idx = name_to_idx.get(dev_name)
            if idx is not None:
                pin_nodes.append((idx, pin_type.upper()))

        if len(pin_nodes) < 2:
            continue

        for i in range(len(pin_nodes)):
            for j in range(i + 1, len(pin_nodes)):
                u_idx, u_pin = pin_nodes[i]
                v_idx, v_pin = pin_nodes[j]
                if u_idx == v_idx:
                    continue

                is_u_sd = (u_pin in ["S", "D"])
                is_v_sd = (v_pin in ["S", "D"])
                weight = 1.0 if (is_u_sd and is_v_sd) else 0.5

                adj_matrix[u_idx, v_idx] = max(
                    adj_matrix[u_idx, v_idx], weight)
                adj_matrix[v_idx, u_idx] = max(
                    adj_matrix[v_idx, u_idx], weight)

    # Self-loops
    for i in range(N):
        adj_matrix[i, i] = 1.0

    # Normalize
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    adj_matrix = adj_matrix / (row_sums + 1e-8)
    return torch.tensor(adj_matrix, dtype=torch.float32)


def load_design_data(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)
    devices = data.get("instances", data.get("devices", []))
    nets = data.get("nets", [])
    # Default grid settings if not present
    grid = data.get("grid", {"poly_pitch": 0.054,
                    "row_pitch": 1.0, "y_nmos": 0.0, "y_pmos": 1.0})
    return devices, nets, grid

###############################################################################
# 2. Neural Networks (GNN + Transformer)
###############################################################################


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        out = torch.bmm(adj, out)
        return F.relu(out)


class GNNEncoder(nn.Module):
    def __init__(self, in_dim: int, hid_dim: int = 64, out_dim: int = GNN_output_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hid_dim)
        self.gcn2 = GCNLayer(hid_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        return h


class PositionEncoder1D(nn.Module):
    def __init__(self, d_model: int, max_len: int = 200):
        super().__init__()
        self.pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        nn.init.normal_(self.pe, std=0.02)

    def forward(self, x: torch.Tensor, step_indices: torch.Tensor) -> torch.Tensor:
        idx = step_indices.squeeze(-1).clamp(0, self.pe.shape[1] - 1)
        return x + self.pe[0][idx].unsqueeze(1)


class TransistorPolicySB3(MaskableActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

        self.N_max = observation_space["node_features"].shape[0]
        feat_dim = observation_space["node_features"].shape[1]

        self.gnn = GNNEncoder(in_dim=5, hid_dim=64, out_dim=GNN_output_dim)
        self.pos_encoder = PositionEncoder1D(
            d_model=GNN_output_dim, max_len=self.N_max + 50)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=GNN_output_dim, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=2)

        self.value_head = nn.Sequential(
            nn.Linear(GNN_output_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.query_token = nn.Parameter(torch.randn(1, 1, GNN_output_dim))
        self._build(lr_schedule)

    def _forward_network(self, obs):
        x = obs["node_features"]
        adj = obs["adj_matrix"]
        step = obs["step_count"].long()

        node_embeds = self.gnn(x, adj)

        # Add positional info to query
        tgt = self.query_token.expand(x.shape[0], 1, -1)
        tgt = self.pos_encoder(tgt, step)

        # Transformer Attention
        dec_out = self.transformer_decoder(tgt, node_embeds).squeeze(1)

        # Action Heads:
        # First N logits -> Normal Placement
        # Second N logits -> Flipped Placement
        logits_nf = torch.bmm(node_embeds, dec_out.unsqueeze(-1)).squeeze(-1)
        logits_f = torch.bmm(node_embeds, dec_out.unsqueeze(-1)).squeeze(-1)

        logits = torch.cat([logits_nf, logits_f], dim=1)  # Shape: [Batch, 2*N]
        return logits, node_embeds

    def _apply_mask(self, logits, action_masks):
        if action_masks is not None:
            if isinstance(action_masks, np.ndarray):
                action_masks = torch.as_tensor(
                    action_masks, device=logits.device)
            logits[~action_masks.bool()] = -1e8
        return logits

    def forward(self, obs, deterministic=False, action_masks=None):
        logits, node_embeds = self._forward_network(obs)
        logits = self._apply_mask(logits, action_masks)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        actions = dist.mode() if deterministic else dist.sample()
        values = self.value_head(node_embeds.mean(dim=1))
        return actions, values, dist.log_prob(actions)

    def evaluate_actions(self, obs, actions, action_masks=None):
        logits, node_embeds = self._forward_network(obs)
        logits = self._apply_mask(logits, action_masks)
        dist = self.action_dist.proba_distribution(action_logits=logits)
        values = self.value_head(node_embeds.mean(dim=1))
        return values, dist.log_prob(actions), dist.entropy()

    def get_distribution(self, obs, action_masks=None):
        logits, _ = self._forward_network(obs)
        logits = self._apply_mask(logits, action_masks)
        return self.action_dist.proba_distribution(action_logits=logits)

    def predict_values(self, obs, action_masks=None):
        _, node_embeds = self._forward_network(obs)
        return self.value_head(node_embeds.mean(dim=1))

###############################################################################
# 3. Environment (Dual Rail + Flip + AutoSave Support)
###############################################################################


class TransistorPlacementEnv(gym.Env):
    def __init__(self, design_data_path: str, n_max_pad: int = 50, reward_cfg: dict = None):
        super().__init__()
        self.devices, self.nets, self.grid = load_design_data(design_data_path)
        self.raw_num_devices = len(self.devices)
        self.N_max = n_max_pad
        self.reward_cfg = reward_cfg if reward_cfg else {}

        # Store Cell Name for Logging
        self.cell_name = pathlib.Path(design_data_path).stem

        # Graph Construction
        self.adj_tensor = build_graph_from_nets(self.devices, self.nets)
        curr_n = self.adj_tensor.shape[0]
        if curr_n < self.N_max:
            pad = self.N_max - curr_n
            self.adj_tensor = F.pad(self.adj_tensor, (0, pad, 0, pad))

        # Feature Construction
        adj_binary = (self.adj_tensor > 0).float()
        degrees = adj_binary.sum(dim=1).numpy()
        self.feat_dim = 5
        self.node_features = np.zeros(
            (self.N_max, self.feat_dim), dtype=np.float32)

        for i, d in enumerate(self.devices):
            d_type = d.get("device_type", d.get("type", "NMOS"))
            t_val = 1.0 if d_type == "PMOS" else 0.0
            w = float(d.get("w", 1e-7)) * 1e6
            l = float(d.get("l", 2e-8)) * 1e6
            nf = float(d.get("nf", 1))
            deg = degrees[i] if i < len(degrees) else 0
            self.node_features[i] = [t_val, w, l, nf, deg]

        # Spaces
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=0, high=1000, shape=(self.N_max, self.feat_dim), dtype=np.float32),
            "adj_matrix":    spaces.Box(low=0, high=1, shape=(self.N_max, self.N_max), dtype=np.float32),
            "step_count":    spaces.Box(low=0, high=self.N_max, shape=(1,), dtype=np.float32),
        })

        # Action: 0~N-1 (Normal), N~2N-1 (Flipped)
        self.action_space = spaces.Discrete(self.N_max * 2)

        # Internal State
        self.placed_mask = np.zeros(self.N_max, dtype=bool)
        self.current_step = 0
        # device_idx -> (x, y, orient, row_type, col_idx, flipped)
        self.positions = {}

        # Dual-Rail Cursors
        self.pmos_next_col = 0
        self.nmos_next_col = 0
        self.pmos_cols = []  # List of (device_idx, flipped_bool)
        self.nmos_cols = []

        self.device_sd_nets = defaultdict(dict)
        self._build_sd_net_map()

        # HPWL Tracking (Incremental)
        self.last_total_hpwl = 0.0

    def _build_sd_net_map(self):
        name_to_idx = {d["name"]: i for i, d in enumerate(self.devices)}
        for net in self.nets:
            valid_tokens = [tok for tok in net[:-1] if _want_pin_token(tok)]
            for tok in valid_tokens:
                dev, pin = tok.split(".", 1)
                idx = name_to_idx.get(dev)
                if idx is not None and pin.upper() in ["S", "D"]:
                    self.device_sd_nets[idx][pin.upper()] = net[-1]

    def action_masks(self) -> np.ndarray:
        # Mask both Normal and Flipped actions for already placed devices
        valid = np.ones(self.N_max * 2, dtype=bool)

        placed_indices = np.where(self.placed_mask)[0]

        # Disable Normal
        valid[placed_indices] = False
        # Disable Flipped
        valid[placed_indices + self.N_max] = False

        # Disable Padding
        if self.raw_num_devices < self.N_max:
            valid[self.raw_num_devices: self.N_max] = False
            valid[self.raw_num_devices + self.N_max:] = False

        return valid

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.placed_mask[:] = False
        self.current_step = 0
        self.positions = {}
        self.pmos_next_col = 0
        self.nmos_next_col = 0
        self.pmos_cols = []
        self.nmos_cols = []
        self.last_total_hpwl = 0.0
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            "node_features": self.node_features,
            "adj_matrix": self.adj_tensor.numpy(),
            "step_count": np.array([self.current_step], dtype=np.float32)
        }

    def _calculate_hpwl_proxy(self):
        """Estimate HPWL based on current placement."""
        net_bbox = {}
        for idx, pos in self.positions.items():
            # pos: (x_val, y_val, orient_str, row_type, col_int, flipped)
            col = pos[4]
            # Differentiate Rows roughly for Y-dist
            row_val = 1.0 if pos[3] == "PMOS" else 0.0

            pins = self.device_sd_nets.get(idx, {})
            for _, net_name in pins.items():
                if _is_power_net(net_name):
                    continue

                if net_name not in net_bbox:
                    net_bbox[net_name] = [col, col, row_val, row_val]
                else:
                    bb = net_bbox[net_name]
                    bb[0] = min(bb[0], col)
                    bb[1] = max(bb[1], col)
                    bb[2] = min(bb[2], row_val)
                    bb[3] = max(bb[3], row_val)

        total = 0.0
        for bb in net_bbox.values():
            w = bb[1] - bb[0]
            # Weight Y distance more to encourage alignment?
            h = (bb[3] - bb[2]) * 5.0
            total += (w + h)
        return total

    def step(self, action):
        action = int(action)

        # Decode Action
        device_idx = action % self.N_max
        flipped = (action >= self.N_max)

        if self.placed_mask[device_idx]:
            return self._get_obs(), -10.0, True, False, {"error": "Invalid"}

        dev = self.devices[device_idx]
        nf = int(dev.get("nf", 1))
        row_type = dev.get("device_type", dev.get("type"))

        self.placed_mask[device_idx] = True

        # --- Placement Logic (Dual Rail) ---
        if row_type == "PMOS":
            start_col = self.pmos_next_col
            self.pmos_cols.append((device_idx, flipped))
            self.pmos_next_col += nf
            y_pos = self.grid["y_pmos"]
            orient = "MX" if not flipped else "MX_R180"  # Example orientation string
        else:
            start_col = self.nmos_next_col
            self.nmos_cols.append((device_idx, flipped))
            self.nmos_next_col += nf
            y_pos = self.grid["y_nmos"]
            orient = "R0" if not flipped else "MY"

        x_pos = start_col * self.grid["poly_pitch"]
        self.positions[device_idx] = (
            x_pos, y_pos, orient, row_type, start_col, flipped)

        # --- Reward Calculation ---
        reward = 0.0
        w_share = self.reward_cfg.get("w_share", 5.0)
        w_break = self.reward_cfg.get("w_break", 2.0)
        w_hpwl = self.reward_cfg.get("w_hpwl", 0.5)

        # 1. Diffusion Sharing Reward
        prev_device_info = None
        if row_type == "PMOS":
            if len(self.pmos_cols) > 1:
                prev_device_info = self.pmos_cols[-2]
        else:
            if len(self.nmos_cols) > 1:
                prev_device_info = self.nmos_cols[-2]

        if prev_device_info:
            prev_idx, prev_flip = prev_device_info

            # Physics: Left Device's Right Pin vs Current Device's Left Pin
            # Pin mapping: Normal (S..D), Flipped (D..S)

            prev_pins = self.device_sd_nets[prev_idx]
            curr_pins = self.device_sd_nets[device_idx]

            # Right pin of prev: D if normal, S if flipped
            net_prev_right = prev_pins.get('S' if prev_flip else 'D')

            # Left pin of curr: S if normal, D if flipped
            net_curr_left = curr_pins.get('D' if flipped else 'S')

            if net_prev_right and net_curr_left and net_prev_right == net_curr_left:
                reward += w_share
            else:
                reward -= w_break

        # 2. Delta HPWL Penalty
        curr_total_hpwl = self._calculate_hpwl_proxy()
        delta_hpwl = curr_total_hpwl - self.last_total_hpwl
        if delta_hpwl > 0:
            reward -= (delta_hpwl * w_hpwl)

        self.last_total_hpwl = curr_total_hpwl

        self.current_step += 1
        done = (self.current_step >= self.raw_num_devices)

        info = {}
        if done:
            # --- Construct Output for Callback ---
            final_width = max(self.pmos_next_col, self.nmos_next_col)

            placement_list = []
            for p_idx, pos_info in self.positions.items():
                # pos_info: (x, y, orient, row, col, flipped)
                p_dev = self.devices[p_idx]
                placement_list.append({
                    "device_name": p_dev["name"],
                    "x": float(pos_info[0]),
                    "y": float(pos_info[1]),
                    "orient": pos_info[2],
                    "row": pos_info[3],
                    "flipped": bool(pos_info[5])
                })

            info["final_metrics"] = {
                "width": final_width,
                "hpwl": curr_total_hpwl,
                "cell_name": self.cell_name,      # Critical for saving
                "placement": placement_list,      # Critical for saving
                "breaks": 0  # Placeholder
            }

        return self._get_obs(), reward, done, False, info

###############################################################################
# 4. Callbacks & BC
###############################################################################


class TensorboardCallback(BaseCallback):
    def __init__(self): super().__init__(0)

    def _on_step(self):
        for info in self.locals['infos']:
            if "final_metrics" in info:
                m = info["final_metrics"]
                self.logger.record("metrics/width", m["width"])
                self.logger.record("metrics/hpwl", m["hpwl"])
        return True


class SaveBestPlacementCallback(BaseCallback):
    """
    Saves the best placement found SO FAR for EACH cell.
    Criteria: Minimize Width, then Minimize HPWL.
    """

    def __init__(self, save_dir: pathlib.Path, verbose=0):
        super().__init__(verbose)
        self.save_dir = save_dir / "best_placements"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.best_records = {}  # cell_name -> (width, hpwl)

    def _on_step(self) -> bool:
        for info in self.locals['infos']:
            if "final_metrics" in info:
                m = info["final_metrics"]
                name = m["cell_name"]
                w = m["width"]
                h = m["hpwl"]

                update = False
                if name not in self.best_records:
                    update = True
                else:
                    bw, bh = self.best_records[name]
                    if w < bw:
                        update = True
                    elif w == bw and h < bh:
                        update = True

                if update:
                    self.best_records[name] = (w, h)
                    if self.verbose > 0:
                        print(f"[Best] {name}: W={w}, HPWL={h:.1f}")
                    self._save_csv(name, m["placement"], w, h)
        return True

    def _save_csv(self, name, placement, w, h):
        fpath = self.save_dir / f"{name}.csv"
        with open(fpath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Cell:", name, "Width:", w, "HPWL:", f"{h:.2f}"])
            writer.writerow(["Name", "X", "Y", "Orient", "Row", "Flipped"])
            for p in placement:
                writer.writerow([
                    p["device_name"], f"{p['x']:.3f}", f"{p['y']:.1f}",
                    p["orient"], p["row"], p["flipped"]
                ])


def pretrain_from_expert_json(model, env_dir, expert_json_path, epochs=30):
    if not os.path.exists(expert_json_path):
        return
    with open(expert_json_path, "r") as f:
        expert_db = json.load(f)

    files = list(env_dir.glob("*.json"))
    trajectories = []

    print(f"--- BC Warm Start ({len(files)} files) ---")

    for fpath in tqdm(files):
        key = fpath.stem
        seq = None
        for k in expert_db:
            if k.startswith(key):
                seq = expert_db[k]["sequence"]
                break
        if not seq:
            continue

        env = TransistorPlacementEnv(str(fpath), n_max_pad=model.policy.N_max)
        name2idx = {d["name"]: i for i, d in enumerate(env.devices)}

        obs = env.reset()[0]
        for name in seq:
            if name not in name2idx:
                continue
            idx = name2idx[name]
            if env.placed_mask[idx]:
                continue

            # Expert data usually doesn't have 'Flip' info easily available
            # We assume NO FLIP for BC (Action = idx) to learn basic order
            obs_c = {k: torch.tensor(v).clone() if isinstance(
                v, np.ndarray) else v for k, v in obs.items()}
            trajectories.append((obs_c, idx))

            obs, _, done, _, _ = env.step(idx)
            if done:
                break

    if not trajectories:
        return

    policy = model.policy
    opt = torch.optim.Adam(policy.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()
    policy.train()

    bs = 64
    for ep in range(epochs):
        random.shuffle(trajectories)
        loss_sum = 0
        for i in range(0, len(trajectories), bs):
            batch = trajectories[i:i+bs]

            obs_batch = defaultdict(list)
            act_batch = []
            for o, a in batch:
                for k, v in o.items():
                    obs_batch[k].append(v)
                act_batch.append(a)

            final_obs = {k: torch.stack(v).to(device)
                         for k, v in obs_batch.items()}
            final_act = torch.tensor(act_batch, dtype=torch.long).to(device)

            dist = policy.get_distribution(final_obs)
            loss = loss_fn(dist.distribution.logits, final_act)

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(batch)

        if (ep+1) % 10 == 0:
            print(f"BC Epoch {ep+1} Loss: {loss_sum/len(trajectories):.4f}")

###############################################################################
# 5. Main
###############################################################################


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--ent-coef", type=float, default=0.01)

    # Curriculum / Rewards
    parser.add_argument("--w-share", type=float, default=5.0)
    parser.add_argument("--w-break", type=float, default=2.0)
    parser.add_argument("--w-hpwl",  type=float, default=0.5)

    parser.add_argument("--resume-from", type=str, default=None)
    args = parser.parse_args()

    reward_cfg = {
        "w_share": args.w_share,
        "w_break": args.w_break,
        "w_hpwl":  args.w_hpwl
    }

    def make_env():
        # Randomly sample a file for each episode
        f = random.choice(list(args.env_dir.glob("*.json")))
        env = TransistorPlacementEnv(
            str(f), n_max_pad=50, reward_cfg=reward_cfg)
        return ActionMasker(env, lambda e: e.action_masks())

    env = DummyVecEnv([make_env])

    model = MaskablePPO(TransistorPolicySB3, env, verbose=1,
                        learning_rate=args.learning_rate, n_steps=args.n_steps,
                        batch_size=args.batch_size, ent_coef=args.ent_coef,
                        device=device, tensorboard_log=str(args.output_dir / "logs"))

    if args.resume_from:
        print(f"Resuming from {args.resume_from}...")
        model.policy.load_state_dict(MaskablePPO.load(
            args.resume_from).policy.state_dict())

    # Pretrain logic (Optional, based on expert_data presence)
    pretrain_from_expert_json(
        model, args.env_dir, "expert_data.json", epochs=50)

    # Callbacks
    callbacks = CallbackList([
        TensorboardCallback(),
        SaveBestPlacementCallback(args.output_dir, verbose=1)
    ])

    print("Starting PPO Training...")
    model.learn(total_timesteps=args.timesteps, callback=callbacks)

    model.save(args.output_dir / "multi_cell_model")
    print("Training Done. Check 'best_placements' folder.")


if __name__ == "__main__":
    main()
