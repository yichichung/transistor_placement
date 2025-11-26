"""
Evaluation script for transistor placement.
Loads a trained model and performs greedy inference on validation cells.

Usage:
    python eval.py --model-path output/multi_cell_model.pth --env-dir val_circuits/ --output-dir eval_results/
"""
from __future__ import annotations
import argparse
import pathlib

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from placement_env import parse_transistor_json, RandomMultiCellEnv
from placement_model import GNNEncoder, TransformerPolicy, ValueNetwork, TransistorPolicySB3, GNN_output_dim
from utils import eval_all_cells_greedy

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained transistor placement model"
    )
    parser.add_argument("--model-path", type=pathlib.Path, required=True,
                        help="Path to trained model .pth file")
    parser.add_argument("--env-dir", type=pathlib.Path, required=True,
                        help="Directory with validation cell JSONs")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("./eval_results"),
                        help="Output directory for evaluation results")

    args = parser.parse_args()

    if not args.model_path.exists():
        parser.error(f"Model file not found: {args.model_path}")

    if not args.env_dir.exists() or not args.env_dir.is_dir():
        parser.error(f"Invalid env-dir: {args.env_dir}")

    print("=" * 80)
    print("EVALUATION MODE: Greedy Inference on All Cells")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Val dir: {args.env_dir}")
    print(f"Output: {args.output_dir}")

    # Load checkpoint
    print("\n[Loading] Reading checkpoint...")
    ckpt = torch.load(args.model_path, map_location=device)
    reward_cfg = ckpt.get("reward_cfg", {
        "w_break": 100.0, "w_dummy": 50.0, "w_share": 10.0,
        "w_hpwl": 2.0, "w_cdist": 5.0
    })

    # Get validation files
    env_files = sorted(list(args.env_dir.glob("*.json")))
    if len(env_files) == 0:
        raise ValueError(f"No JSON files found in {args.env_dir}")

    print(f"\nFound {len(env_files)} cells to evaluate")
    for f in env_files[:5]:
        print(f"  - {f.stem}")
    if len(env_files) > 5:
        print(f"  ... and {len(env_files) - 5} more")

    # Build networks
    print("\n[Building] Constructing neural networks...")
    first_data = parse_transistor_json(env_files[0], verbose=False)
    feature_dim = first_data["features"].size(1)

    encoder = GNNEncoder(in_dim=feature_dim, out_dim=GNN_output_dim).to(device)
    policy_net = TransformerPolicy(embed_dim=GNN_output_dim).to(device)
    value_net = ValueNetwork(embed_dim=GNN_output_dim, extra_dim=5).to(device)

    # Load weights
    print("[Loading] Loading model weights...")
    encoder.load_state_dict(ckpt["encoder"])
    policy_net.load_state_dict(ckpt["policy"])
    value_net.load_state_dict(ckpt["value"])

    encoder.eval()
    policy_net.eval()
    value_net.eval()

    print("✓ Model loaded successfully")

    # Create environment and model
    print("\n[Environment] Creating multi-cell environment...")

    def _make_env():
        return RandomMultiCellEnv(env_files, reward_cfg, device=device)

    env = DummyVecEnv([_make_env])

    policy_kwargs = dict(
        encoder=encoder,
        policy_net=policy_net,
        value_net=value_net,
        graph_data=first_data,
        env_ref=env.envs[0]
    )

    model = PPO(
        policy=TransistorPolicySB3,
        env=env,
        policy_kwargs=policy_kwargs,
        verbose=0,
        device=device
    )

    # Run evaluation
    out_dir = args.output_dir / "placements"
    print(f"\n[Evaluation] Starting greedy inference...\n")
    eval_all_cells_greedy(env, model, out_dir, device=str(device))

    print(f"\n{'='*80}")
    print(f"✅ Evaluation Complete!")
    print(f"{'='*80}")
    print(f"Results saved to: {out_dir}")

    # List generated files
    result_files = sorted(list(out_dir.glob("*_best_placement.csv")))
    if result_files:
        print(f"\nGenerated {len(result_files)} placement files:")
        for f in result_files[:10]:
            print(f"  - {f.name}")
        if len(result_files) > 10:
            print(f"  ... and {len(result_files) - 10} more")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
