#!/usr/bin/env python3
"""Run a trained PM01 AMP policy in IsaacLab 2.3.1."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "amp_mjlab_pm01"))
sys.path.insert(0, str(ROOT / "third_party"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Play an AMP_mjlab PM01 checkpoint")
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task", default="AMP-MJLAB-PM01-Phase2-Play-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=0, help="Stop after N policy steps; zero runs until the app closes.")
parser.add_argument("--real_time", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import AmpOnPolicyRunner

import amp_mjlab_pm01.tasks  # noqa: F401


def main() -> None:
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")

    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or agent_cfg.device
    agent_cfg.device = env_cfg.sim.device

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint), load_optimizer=False)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[INFO] Loaded checkpoint: {checkpoint}")

    obs = env.get_observations()
    step = 0
    while simulation_app.is_running() and (args_cli.steps <= 0 or step < args_cli.steps):
        started = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        step += 1
        if args_cli.real_time:
            remaining = env.unwrapped.step_dt - (time.time() - started)
            if remaining > 0:
                time.sleep(remaining)
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
