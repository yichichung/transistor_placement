"""
Training callbacks for monitoring and saving best results.
"""
from __future__ import annotations
import csv
import pathlib
from typing import Dict, List, Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


class TqdmCallback(BaseCallback):
    """Progress bar callback using tqdm."""
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
        if self.pbar and delta > 0:
            self.pbar.update(delta)
            self._last_update = current
        return True

    def _on_training_end(self) -> None:
        if self.pbar:
            self.pbar.close()


class BestPerCellCallback(BaseCallback):
    """
    Tracks best placement for each cell using weighted scoring.
    Saves best placements to {out_dir}/{cell_name}_best_placement.csv
    """
    def __init__(self, env, out_dir: pathlib.Path, reward_cfg: Dict = None, verbose: int = 1):
        super().__init__(verbose)
        self.env = env
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Track best result per cell: cell_name -> {"score": float, "metrics": dict, "path": str}
        self.best_by_cell: Dict[str, Dict[str, Any]] = {}
        self.episode_count = 0
        self.reward_cfg = reward_cfg

    def _get_weights(self) -> Dict[str, float]:
        """Get reward weights (priority: passed param > model.reward_cfg > defaults)."""
        if self.reward_cfg is not None:
            return self.reward_cfg

        if hasattr(self.model, "reward_cfg") and self.model.reward_cfg:
            return self.model.reward_cfg

        # Default values
        return {
            "w_break": 100.0,
            "w_dummy": 50.0,
            "w_share": 10.0,
            "w_hpwl": 2.0,
            "w_cdist": 5.0
        }

    def _compute_weighted_score(self, metrics: Dict[str, float]) -> float:
        """
        Compute weighted score (lower is better).
        Score = w_break*breaks + w_dummy*dummy - w_share*shared + w_hpwl*hpwl + w_cdist*col_dist
        """
        weights = self._get_weights()
        score = (
            weights.get("w_break", 100.0) * metrics.get("breaks", 0.0)
            + weights.get("w_dummy", 50.0) * metrics.get("dummy", 0.0)
            - weights.get("w_share", 10.0) * metrics.get("shared", 0.0)
            + weights.get("w_hpwl", 2.0) * metrics.get("hpwl", 0.0)
            + weights.get("w_cdist", 5.0) * metrics.get("col_dist", 0.0)
        )
        return float(score)

    def _cell_name_from_info(self, info: Dict[str, Any]) -> str:
        """Extract cell name from info or environment."""
        # 1. Priority: from info
        if "cell_name" in info and info["cell_name"] and info["cell_name"] != "unknown_cell":
            return info["cell_name"]

        # 2. From environment
        try:
            env_single = self.env.envs[0]
            if hasattr(env_single, "graph") and "cell_name" in env_single.graph:
                cell_name = env_single.graph["cell_name"]
                if cell_name and cell_name != "unknown_cell":
                    return cell_name
        except Exception:
            pass

        # 3. Fallback
        return f"cell_{self.episode_count}"

    def _write_csv(self, csv_path: pathlib.Path, placement: List[Dict[str, Any]]) -> None:
        """Write placement to CSV file."""
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            fieldnames = ["device_name", "device_type", "row", "column",
                          "x", "y", "orient", "w", "l", "nf", "pair_with"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(placement)

    def _maybe_update_cell_best(self, cell_name: str, metrics: Dict[str, float],
                                placement: List[Dict[str, Any]]):
        """Check and update best result for this cell."""
        new_score = self._compute_weighted_score(metrics)

        rec = self.best_by_cell.get(cell_name)

        # Update if first time or new score is better (lower)
        should_update = (rec is None) or (new_score < rec["score"])

        if should_update:
            final_path = self.out_dir / f"{cell_name}_best_placement.csv"
            self._write_csv(final_path, placement)

            self.best_by_cell[cell_name] = {
                "score": new_score,
                "metrics": metrics.copy(),
                "path": str(final_path)
            }

            if self.verbose:
                print(f"[Best] {cell_name}: score={new_score:.2f} "
                      f"(breaks={metrics.get('breaks', 0)}, "
                      f"dummy={metrics.get('dummy', 0)}, "
                      f"shared={metrics.get('shared', 0):.1f}, "
                      f"hpwl={metrics.get('hpwl', 0):.3f}, "
                      f"col_dist={metrics.get('col_dist', 0):.2f})")

    def _on_step(self) -> bool:
        try:
            # Get dones and infos
            dones = self.locals.get("dones", [False])
            infos = self.locals.get("infos", [{}])

            # Ensure dones is iterable
            if not isinstance(dones, (list, tuple, np.ndarray)):
                dones = [dones]

            # Check if any episode completed
            if not any(dones):
                return True

            # Check infos validity
            if not infos or len(infos) == 0:
                if self.verbose > 1:
                    print("[Warning] No infos available in callback")
                return True

            info = infos[0]

            # Check for required fields
            if "final_metrics" not in info:
                if self.verbose > 1:
                    print("[Warning] No final_metrics in info")
                return True

            if "placement" not in info:
                if self.verbose > 1:
                    print("[Warning] No placement in info")
                return True

            self.episode_count += 1
            metrics = info["final_metrics"]
            cell_name = self._cell_name_from_info(info)
            placement = info["placement"]

            # TensorBoard logging (improved: use cell-specific keys)
            if hasattr(self.model, "logger") and self.model.logger:
                # Global statistics
                self.model.logger.record(
                    "placement/breaks", float(metrics.get("breaks", 0)))
                self.model.logger.record(
                    "placement/dummy", float(metrics.get("dummy", 0)))
                self.model.logger.record(
                    "placement/shared", float(metrics.get("shared", 0)))
                self.model.logger.record(
                    "placement/hpwl", float(metrics.get("hpwl", 0)))
                self.model.logger.record(
                    "placement/col_dist", float(metrics.get("col_dist", 0)))

                # Weighted score
                score = self._compute_weighted_score(metrics)
                self.model.logger.record("placement/weighted_score", score)

                # Per-cell statistics
                self.model.logger.record(
                    f"placement/{cell_name}/breaks", float(metrics.get("breaks", 0)))
                self.model.logger.record(
                    f"placement/{cell_name}/dummy", float(metrics.get("dummy", 0)))
                self.model.logger.record(
                    f"placement/{cell_name}/shared", float(metrics.get("shared", 0)))
                self.model.logger.record(
                    f"placement/{cell_name}/hpwl", float(metrics.get("hpwl", 0)))
                self.model.logger.record(
                    f"placement/{cell_name}/weighted_score", score)

            # Write last placement
            last_path = self.out_dir / f"{cell_name}_last_placement.csv"
            self._write_csv(last_path, placement)

            # Update best placement
            if placement:
                self._maybe_update_cell_best(cell_name, metrics, placement)

            # Debug output
            if self.verbose > 1:
                print(f"[Callback] Episode {self.episode_count}: "
                      f"cell='{cell_name}', placement_len={len(placement)}")

        except Exception as e:
            if self.verbose:
                import traceback
                print(f"[Error] BestPerCellCallback exception:")
                print(traceback.format_exc())

        return True
