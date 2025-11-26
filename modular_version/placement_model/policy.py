"""
Neural network architectures for transistor placement.
Includes GNN encoder, Transformer policy, Value network, and SB3 integration.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, Tuple, Union, Set, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy


# Global constant
GNN_output_dim = 128


###############################################################################
# GNN Encoder
###############################################################################

class GCNLayer(nn.Module):
    """Graph Convolutional Layer."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return F.relu(self.linear(torch.matmul(adj, x)))


class GNNEncoder(nn.Module):
    """
    Graph Neural Network encoder using stacked GCN layers.
    Encodes device connectivity into node embeddings.
    """
    def __init__(self, in_dim: int, hid_dim: int = 64, out_dim: int = GNN_output_dim):
        super().__init__()
        self.gcn1 = GCNLayer(in_dim, hid_dim)
        self.gcn2 = GCNLayer(hid_dim, hid_dim)
        self.gcn3 = GCNLayer(hid_dim, out_dim)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Node features [N, in_dim]
            adj: Adjacency matrix [N, N]

        Returns:
            Node embeddings [N, out_dim]
        """
        h = self.gcn1(x, adj)
        h = self.gcn2(h, adj)
        h = self.gcn3(h, adj)
        return h


###############################################################################
# Transformer Policy
###############################################################################

class PositionEncoder1D(nn.Module):
    """1D positional encoding with optional Fourier features."""
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
    """Cross-attention block with feedforward network."""
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
    """
    Transformer-based policy network.
    Uses cross-attention to attend to placed devices and predict scores.
    """
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
        """Build attention bias based on column distance."""
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
                placed_indices: Optional[Union[Set, List, np.ndarray]] = None,
                node_positions: Optional[Dict[int, Tuple]] = None,
                next_cols: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            cell_embeddings: Node embeddings [N, embed_dim]
            placed_indices: Indices of already placed devices
            node_positions: Position dictionary for placed devices
            next_cols: Next column indices for positional bias

        Returns:
            Action scores [N]
        """
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
# Value Network
###############################################################################

class ValueNetwork(nn.Module):
    """Value function network for PPO."""
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
# SB3 Policy Integration
###############################################################################

class TransistorPolicySB3(ActorCriticPolicy):
    """
    Custom Stable-Baselines3 policy integrating GNN and Transformer.
    Implements action masking and value prediction.
    """
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
        """Compute GNN embeddings for current graph."""
        device = next(self.encoder.parameters()).device
        features = self.env_ref.graph["features"].to(device)
        adj = self.env_ref.graph["adj"].to(device)
        return self.encoder(features, adj)

    def _build_next_cols(self, embeddings):
        """Build column indices tensor for all nodes."""
        N = embeddings.size(0)
        col_idx = self.env_ref.col_idx
        next_cols = torch.full((N,), float(col_idx),
                               device=embeddings.device, dtype=embeddings.dtype)
        return next_cols

    def _get_state_embedding(self, embeddings):
        """Compute state embedding for value function."""
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
        """
        Forward pass with action masking.

        Args:
            obs: Observation tensor
            deterministic: Use greedy action selection

        Returns:
            actions, values, log_probs
        """
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

            N = self.env_ref.N
            N_max = self.env_ref.N_max

            mask = torch.ones(N, dtype=torch.bool, device=scores.device)
            for idx in range(N):
                if self.env_ref.placed[idx] == 1:
                    mask[idx] = False

            if scores.numel() < N_max:
                scores = torch.cat([
                    scores,
                    torch.full((N_max - scores.numel(),), -1e9,
                               device=scores.device, dtype=scores.dtype)
                ], dim=0)

            full_mask = torch.zeros(
                N_max, dtype=torch.bool, device=scores.device)
            full_mask[:N] = mask
            scores = scores.masked_fill(~full_mask, -1e9)

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
        """Prediction interface for SB3."""
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions.view(-1)

    def evaluate_actions(self, obs, actions):
        """Evaluate actions for PPO update."""
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

            N = self.env_ref.N
            N_max = self.env_ref.N_max

            mask = torch.ones(N, dtype=torch.bool, device=scores.device)
            for idx in range(N):
                if self.env_ref.placed[idx] == 1:
                    mask[idx] = False

            if scores.numel() < N_max:
                scores = torch.cat([
                    scores,
                    torch.full((N_max - scores.numel(),), -1e9,
                               device=scores.device, dtype=scores.dtype)
                ], dim=0)

            full_mask = torch.zeros(
                N_max, dtype=torch.bool, device=scores.device)
            full_mask[:N] = mask
            scores = scores.masked_fill(~full_mask, -1e9)

            dist = torch.distributions.Categorical(logits=scores)
            log_probs.append(dist.log_prob(actions[i]))
            entropies.append(dist.entropy())

            state_embed = self._get_state_embedding(embeddings)
            values.append(self.value_net_model(state_embed))

        return torch.stack(values), torch.stack(log_probs), torch.stack(entropies)

    def predict_values(self, obs):
        """Predict state values."""
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
