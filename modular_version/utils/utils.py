"""
Utility functions for evaluation and result export.
"""
from __future__ import annotations
import csv
import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from stable_baselines3.common.vec_env import DummyVecEnv


def eval_all_cells_greedy(env: 'DummyVecEnv', model, out_dir: pathlib.Path, device: str = "cpu"):
    """
    Evaluate all cells using greedy inference.

    For each cell in the environment:
    1. Reset to that cell
    2. Run greedy policy until done
    3. Export placement to {cell_name}_best_placement.csv

    Args:
        env: Vectorized environment (wrapping RandomMultiCellEnv)
        model: Trained PPO model
        out_dir: Output directory for results
        device: Device for computation
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    mc_env = env.envs[0]  # RandomMultiCellEnv (single environment)
    env_files = list(mc_env.env_files)  # All JSON files

    for p in env_files:
        # 1) Rebuild for this cell
        g = mc_env._graphs[str(p)]
        mc_env._rebuild_from_dict(g)
        obs, info = mc_env.reset()  # SB3 returns (obs, info)
        done = False

        # 2) Loop: greedy actions until done
        while not done:
            # If custom policy already does masking in forward, use deterministic=True
            try:
                action, _ = model.predict(obs, deterministic=True)
            except Exception:
                # If model.predict incompatible, switch to manual scoring + masking path
                raise RuntimeError(
                    "model.predict failed. Consider manual scoring + masking approach.")

            obs, reward, terminated, truncated, info = mc_env.step(action)
            done = bool(terminated or truncated)

        # 3) Episode complete: export CSV (one per cell)
        placement = mc_env.get_current_placement_dicts()
        cell_name = g.get("cell_name", pathlib.Path(p).stem)
        csv_path = out_dir / f"{cell_name}_best_placement.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "device_name", "device_type", "row", "column", "x", "y", "orient", "w", "l", "nf", "pair_with"
            ])
            writer.writeheader()
            writer.writerows(placement)
        print(f"[eval-all] wrote {csv_path}")
