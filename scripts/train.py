#!/usr/bin/env python3
"""Train the PM01 AMP policy in IsaacLab 2.3.1."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "amp_mjlab_pm01"))
sys.path.insert(0, str(ROOT / "third_party"))

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="AMP_mjlab PM01 training")
parser.add_argument("--task", default="AMP-MJLAB-PM01-Phase1-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--run_name", default="")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--resume", type=Path, default=None, help="Checkpoint to resume from.")
parser.add_argument(
    "--finetune",
    action="store_true",
    help="Load model/normalizer weights but start a fresh optimizer and iteration count in the new run.",
)
parser.add_argument(
    "--expand_observation",
    type=int,
    default=0,
    help=(
        "For --finetune only: append this many zero-initialized observation columns to "
        "the actor/critic first layers and empirical normalizers."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch
from datetime import datetime

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from rsl_rl.runners import AmpOnPolicyRunner

import amp_mjlab_pm01.tasks  # noqa: F401


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main() -> None:
    if args_cli.finetune and args_cli.resume is None:
        raise ValueError("--finetune requires --resume CHECKPOINT")
    if args_cli.expand_observation and not args_cli.finetune:
        raise ValueError("--expand_observation requires --finetune")
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.seed is not None:
        agent_cfg.seed = args_cli.seed
    if args_cli.run_name:
        agent_cfg.run_name = args_cli.run_name

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device or agent_cfg.device
    agent_cfg.device = env_cfg.sim.device

    log_root = ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    run = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run += f"_{agent_cfg.run_name}"
    log_dir = log_root / run
    print(f"[INFO] Logging experiment in: {log_dir}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    if not isinstance(env.unwrapped, ManagerBasedRLEnv):
        raise TypeError(f"Expected ManagerBasedRLEnv, got {type(env.unwrapped)}")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = AmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=str(log_dir), device=agent_cfg.device)
    print("[INFO] Runner initialized", flush=True)
    if args_cli.resume is not None:
        checkpoint = args_cli.resume.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        if args_cli.expand_observation:
            _load_expanded_observation_finetune(runner, checkpoint, args_cli.expand_observation)
        else:
            runner.load(str(checkpoint), load_optimizer=not args_cli.finetune)
        if args_cli.finetune:
            runner.current_learning_iteration = 0
            print(
                f"[INFO] Fine-tuning weights and normalizer from: {checkpoint}; "
                "optimizer and iteration count are fresh",
                flush=True,
            )
        else:
            print(f"[INFO] Resumed checkpoint: {checkpoint}", flush=True)
    runner.add_git_repo_to_log(__file__)
    params_dir = log_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    for name, cfg in (("env", env_cfg), ("agent", agent_cfg)):
        with (params_dir / f"{name}.json").open("w", encoding="utf-8") as handle:
            json.dump(cfg.to_dict(), handle, indent=2, default=str)
    print("[INFO] Configuration snapshots saved; entering training", flush=True)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print("[INFO] Training returned", flush=True)
    env.close()


def _expand_last_dimension(source: torch.Tensor, target: torch.Tensor, extra: int, key: str) -> torch.Tensor:
    """Copy a checkpoint tensor into an observation-expanded target tensor."""
    if source.ndim != target.ndim or source.shape[:-1] != target.shape[:-1]:
        raise ValueError(f"Cannot expand {key}: source={tuple(source.shape)}, target={tuple(target.shape)}")
    if source.shape[-1] + extra != target.shape[-1]:
        raise ValueError(f"Unexpected expanded size for {key}: source={tuple(source.shape)}, target={tuple(target.shape)}")
    expanded = target.detach().clone()
    expanded[..., : source.shape[-1]] = source
    return expanded


def _load_expanded_observation_finetune(runner, checkpoint: Path, extra: int) -> None:
    """Load a base policy after appending explicit observation features."""
    loaded = torch.load(checkpoint, weights_only=False, map_location=runner.device)
    state = loaded["model_state_dict"]
    target = runner.alg.policy.state_dict()
    for key in ("actor.0.weight", "critic.0.weight"):
        if key not in state or key not in target:
            raise KeyError(f"Checkpoint/model is missing {key}")
        state[key] = _expand_last_dimension(state[key], target[key], extra, key)
    runner.alg.policy.load_state_dict(state)
    if "discriminator_state_dict" in loaded:
        runner.alg.discriminator.load_state_dict(loaded["discriminator_state_dict"])
    if "amp_normalizer" in loaded:
        runner.alg.amp_normalizer = loaded["amp_normalizer"]
    if runner.empirical_normalization:
        for normalizer, key in ((runner.obs_normalizer, "obs_norm_state_dict"), (runner.privileged_obs_normalizer, "privileged_obs_norm_state_dict")):
            source = loaded[key]
            target_state = normalizer.state_dict()
            expanded = {
                name: (_expand_last_dimension(value, target_state[name], extra, f"{key}.{name}")
                       if isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[-1] + extra == target_state[name].shape[-1]
                       else value)
                for name, value in source.items()
            }
            normalizer.load_state_dict(expanded)
    print(f"[INFO] Expanded-observation fine-tune loaded from: {checkpoint}; appended {extra} zero-initialized observation feature(s)", flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
