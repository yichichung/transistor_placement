"""
Training script for transistor placement using GNN + Transformer + PPO.

Usage:
    # Single cell training
    python train.py --input-file circuits/cell1.json --timesteps 100000

    # Multi-cell training
    python train.py --env-dir circuits/ --timesteps 500000

    # Resume from checkpoint
    python train.py --env-dir circuits/ --resume-from output/multi_cell_model.pth
"""
from __future__ import annotations
import argparse
import pathlib
from typing import Dict, Optional

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from placement_env import parse_transistor_json, TransistorPlacementEnv, RandomMultiCellEnv
from placement_model import GNNEncoder, TransformerPolicy, ValueNetwork, TransistorPolicySB3, GNN_output_dim
from utils import TqdmCallback, BestPerCellCallback

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_sb3_model(graph_data, reward_cfg, device, ppo_kwargs=None, tb_log=None):
    """Create PPO model for single-cell training."""
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
        tensorboard_log=tb_log
    )

    model.reward_cfg = reward_cfg
    return model, env, (encoder, policy_net, value_net)


def make_multicell_model(env_files, reward_cfg, device, ppo_kwargs=None, tb_log=None):
    """Create PPO model for multi-cell training."""
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
        tensorboard_log=tb_log
    )

    model.reward_cfg = reward_cfg
    return model, env, (encoder, policy_net, value_net)


def train_transistor_placement(
    json_path: Optional[pathlib.Path] = None,
    env_dir: Optional[pathlib.Path] = None,
    output_dir: pathlib.Path = pathlib.Path("./output"),
    total_timesteps: int = 100_000,
    reward_cfg: Optional[Dict] = None,
    ppo_kwargs: Optional[Dict] = None,
    device_arg=None,
    resume_from: Optional[pathlib.Path] = None
):
    """
    Main training function.

    Args:
        json_path: Path to single cell JSON file
        env_dir: Directory with multiple cell JSON files
        output_dir: Output directory for models and results
        total_timesteps: Total training timesteps
        reward_cfg: Reward weight configuration
        ppo_kwargs: PPO hyperparameters
        device_arg: Device override
        resume_from: Path to checkpoint for resuming training
    """
    device_final = device_arg or device
    print(f"Using device: {device_final}")
    if device_final.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    if reward_cfg is None:
        reward_cfg = {
            "w_break": 100.0,
            "w_dummy": 50.0,
            "w_share": 10.0,
            "w_hpwl": 2.0,
            "w_cdist": 5.0
        }

    print(f"\n[Reward Weights]")
    for k, v in reward_cfg.items():
        print(f"  {k}: {v}")

    output_dir.mkdir(parents=True, exist_ok=True)
    tb_log_dir = str(output_dir / "tensorboard")

    if env_dir is not None:
        print(f"\n[Multi-Cell Mode] Loading cells from: {env_dir}")
        env_files = sorted(list(env_dir.glob("*.json")))
        if len(env_files) == 0:
            raise ValueError(f"No JSON files found in {env_dir}")

        print(f"  Found {len(env_files)} cells")
        model, env, networks = make_multicell_model(
            env_files, reward_cfg, device_final, ppo_kwargs, tb_log_dir)
        output_name = "multi_cell"
    else:
        print(f"\n[Single-Cell Mode] Loading: {json_path}")
        graph_data = parse_transistor_json(json_path, verbose=True)
        graph_data["features"] = graph_data["features"].to(device_final)
        graph_data["adj"] = graph_data["adj"].to(device_final)

        model, env, networks = make_sb3_model(
            graph_data, reward_cfg, device_final, ppo_kwargs, tb_log_dir)
        output_name = json_path.stem

    encoder, policy_net, value_net = networks

    # Resume from checkpoint if provided
    if resume_from is not None and resume_from.exists():
        print(f"\n[Resume] Loading weights from {resume_from}")
        try:
            ckpt = torch.load(resume_from, map_location=device_final)
            if "encoder" in ckpt:
                encoder.load_state_dict(ckpt["encoder"])
                print("  ✓ Encoder loaded")
            if "policy" in ckpt:
                policy_net.load_state_dict(ckpt["policy"])
                print("  ✓ Policy loaded")
            if "value" in ckpt:
                value_net.load_state_dict(ckpt["value"])
                print("  ✓ Value loaded")
        except Exception as e:
            print(f"  ✗ Resume failed: {e}")
            print("  Starting from scratch...")

    total_params = sum(p.numel() for p in encoder.parameters()) + \
        sum(p.numel() for p in policy_net.parameters()) + \
        sum(p.numel() for p in value_net.parameters())
    print(f"\n[Model] Total parameters: {total_params:,}")

    print(f"\n[Training] Starting {total_timesteps} timesteps...")
    print(f"  TensorBoard: tensorboard --logdir {tb_log_dir}")

    best_dir = output_dir / "best_by_cell"

    callbacks = [
        TqdmCallback(total_timesteps),
        BestPerCellCallback(env, best_dir, reward_cfg=reward_cfg, verbose=1),
    ]

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=False,
        tb_log_name=output_name
    )

    # Save model
    model_path = output_dir / f"{output_name}_model.pth"
    torch.save({
        "encoder": encoder.state_dict(),
        "policy": policy_net.state_dict(),
        "value": value_net.state_dict(),
        "reward_cfg": reward_cfg
    }, model_path)

    # Training summary
    print(f"\n{'='*80}")
    print(f"[Training Complete]")
    print(f"{'='*80}")
    print(f"  Model saved: {model_path}")
    print(f"  TensorBoard logs: {tb_log_dir}")
    print(f"  Best placements: {best_dir}")

    # List saved best placements
    best_files = sorted(list(best_dir.glob("*_best_placement.csv")))
    if best_files:
        print(f"\n  Found {len(best_files)} best placement files:")
        for f in best_files[:10]:
            print(f"    - {f.name}")
        if len(best_files) > 10:
            print(f"    ... and {len(best_files) - 10} more")
    else:
        print(f"\n  ⚠️  Warning: No best placement files found!")
        print(f"     Check if episodes completed successfully.")

    # Callback statistics
    best_callback = None
    for cb in callbacks:
        if isinstance(cb, BestPerCellCallback):
            best_callback = cb
            break

    if best_callback:
        print(f"\n  Callback Statistics:")
        print(f"    - Total episodes completed: {best_callback.episode_count}")
        print(
            f"    - Unique cells with best placement: {len(best_callback.best_by_cell)}")
        if best_callback.best_by_cell:
            print(f"\n  Best scores by cell:")
            for cell_name, rec in sorted(best_callback.best_by_cell.items())[:5]:
                print(f"    - {cell_name}: score={rec['score']:.2f}")
            if len(best_callback.best_by_cell) > 5:
                print(
                    f"    ... and {len(best_callback.best_by_cell) - 5} more")

    print(f"{'='*80}\n")

    return model, env


def main():
    parser = argparse.ArgumentParser(
        description="Transistor Placement: GNN + Transformer + PPO"
    )
    parser.add_argument("--input-file", type=pathlib.Path, default=None,
                        help="Single cell JSON file")
    parser.add_argument("--env-dir", type=pathlib.Path, default=None,
                        help="Directory with multiple cell JSONs")
    parser.add_argument("--timesteps", type=int, default=100_000,
                        help="Total training timesteps")
    parser.add_argument("--output-dir", type=pathlib.Path,
                        default=pathlib.Path("./output"),
                        help="Output directory")
    parser.add_argument("--w-break", type=float, default=100.0,
                        help="Weight for diffusion breaks")
    parser.add_argument("--w-dummy", type=float, default=50.0,
                        help="Weight for dummy transistors")
    parser.add_argument("--w-share", type=float, default=10.0,
                        help="Weight for shared diffusions")
    parser.add_argument("--w-hpwl", type=float, default=2.0,
                        help="Weight for HPWL")
    parser.add_argument("--w-cdist", type=float, default=5.0,
                        help="Weight for pair column distance")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
                        help="PPO learning rate")
    parser.add_argument("--n-steps", type=int, default=2048,
                        help="PPO rollout steps")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="PPO batch size")
    parser.add_argument("--ent-coef", type=float, default=0.02,
                        help="PPO entropy coefficient")
    parser.add_argument("--resume-from", type=pathlib.Path, default=None,
                        help="Path to checkpoint .pth file to resume training")

    args = parser.parse_args()

    if args.input_file is None and args.env_dir is None:
        parser.error("Must provide either --input-file or --env-dir")

    reward_cfg = {
        "w_break": args.w_break,
        "w_dummy": args.w_dummy,
        "w_share": args.w_share,
        "w_hpwl": args.w_hpwl,
        "w_cdist": args.w_cdist,
    }

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
    print("=" * 80)

    if args.env_dir is not None:
        train_transistor_placement(
            json_path=None,
            env_dir=args.env_dir,
            output_dir=args.output_dir,
            total_timesteps=args.timesteps,
            reward_cfg=reward_cfg,
            ppo_kwargs=ppo_kwargs,
            device_arg=device,
            resume_from=args.resume_from
        )
    else:
        train_transistor_placement(
            json_path=args.input_file,
            env_dir=None,
            output_dir=args.output_dir,
            total_timesteps=args.timesteps,
            reward_cfg=reward_cfg,
            ppo_kwargs=ppo_kwargs,
            device_arg=device,
            resume_from=args.resume_from
        )


if __name__ == "__main__":
    main()
