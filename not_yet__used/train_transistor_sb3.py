
# train_transistor_sb3.py
# Train PPO on TransistorPlacementEnv using Stable-Baselines3.
# If you have a custom Transformer policy/agent from the chip-level reference,
# you can swap "MlpPolicy" with your PlacementPolicySB3 and pass policy_kwargs.

from typing import Dict, Any
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from parser_transistor import parse_transistor_json
from env_transistor import TransistorPlacementEnv

def make_env(json_path: str, reward_cfg: Dict[str, float]):
    graph = parse_transistor_json(json_path)
    def _fn():
        return TransistorPlacementEnv(graph, reward_cfg)
    return _fn

def train(json_path: str, total_timesteps: int = 50_000, policy="MlpPolicy", policy_kwargs: Dict[str,Any]=None):
    reward_cfg = dict(w_break=10.0, w_share=4.0, w_dummy_eff=0.0, w_hpwl=0.5)
    env = DummyVecEnv([make_env(json_path, reward_cfg)])
    model = PPO(policy=policy, env=env, n_steps=2048, batch_size=64, learning_rate=3e-4,
                policy_kwargs=policy_kwargs or {})
    model.learn(total_timesteps=total_timesteps)
    return model
