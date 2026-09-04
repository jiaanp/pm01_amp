from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .terminations import DelayedTerminationManager


@dataclass
class _MotionFrames:
    root_pos: torch.Tensor
    root_quat: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor


def _load_motion_dir(path: str, device: str, target_joint_names: list[str]) -> _MotionFrames:
    root_pos = []
    root_quat = []
    root_lin_vel = []
    root_ang_vel = []
    joint_pos = []
    joint_vel = []
    files = sorted(name for name in os.listdir(path) if name.endswith(".npz"))
    if not files:
        raise RuntimeError(f"No NPZ motions in {path}")
    for name in files:
        data = np.load(os.path.join(path, name), allow_pickle=False)
        source_joint_names = [str(value) for value in data["joint_names"]]
        order = [source_joint_names.index(name) for name in target_joint_names]
        body_names = [str(value) for value in data["body_names"]]
        root_id = body_names.index("LINK_BASE")
        root_pos.append(torch.as_tensor(data["body_pos_w"][:, root_id], dtype=torch.float32, device=device))
        root_quat.append(torch.as_tensor(data["body_quat_w"][:, root_id], dtype=torch.float32, device=device))
        root_lin_vel.append(torch.as_tensor(data["body_lin_vel_w"][:, root_id], dtype=torch.float32, device=device))
        root_ang_vel.append(torch.as_tensor(data["body_ang_vel_w"][:, root_id], dtype=torch.float32, device=device))
        joint_pos.append(torch.as_tensor(data["joint_pos"][:, order], dtype=torch.float32, device=device))
        joint_vel.append(torch.as_tensor(data["joint_vel"][:, order], dtype=torch.float32, device=device))
    return _MotionFrames(*(torch.cat(values, dim=0) for values in (
        root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel
    )))


class MotionResetManager:
    _instances: dict[str, "MotionResetManager"] = {}

    def __init__(self, walk: _MotionFrames, recovery: _MotionFrames | None):
        self.walk = walk
        self.recovery = recovery

    @classmethod
    def initialize(cls, env, motion_dir: str, recovery_dir: str | None):
        asset: Articulation = env.scene["robot"]
        cls._instances[motion_dir] = cls(
            _load_motion_dir(motion_dir, env.device, list(asset.joint_names)),
            _load_motion_dir(recovery_dir, env.device, list(asset.joint_names)) if recovery_dir else None,
        )

    @classmethod
    def get(cls, motion_dir: str) -> "MotionResetManager":
        return cls._instances[motion_dir]

    def reset(self, env, env_ids: torch.Tensor, motion_dir: str, asset_cfg: SceneEntityCfg):
        delay_mask = getattr(env.termination_manager, "_delay_env_mask", None)
        if delay_mask is None:
            self._write(env, env_ids, self.walk, asset_cfg)
            return
        is_delay = delay_mask[env_ids]
        normal_ids = env_ids[~is_delay]
        delay_ids = env_ids[is_delay]
        if len(normal_ids):
            self._write(env, normal_ids, self.walk, asset_cfg)
        if len(delay_ids):
            self._write(env, delay_ids, self.recovery or self.walk, asset_cfg)

    @staticmethod
    def _write(env, env_ids: torch.Tensor, frames: _MotionFrames, asset_cfg: SceneEntityCfg):
        asset: Articulation = env.scene[asset_cfg.name]
        indices = torch.randint(0, frames.root_pos.shape[0], (len(env_ids),), device=env.device)
        position = env.scene.env_origins[env_ids].clone()
        position[:, 2] += frames.root_pos[indices, 2]
        root_pose = torch.cat((position, frames.root_quat[indices]), dim=-1)
        root_velocity = torch.cat((frames.root_lin_vel[indices], frames.root_ang_vel[indices]), dim=-1)
        joint_pos = frames.joint_pos[indices]
        joint_vel = frames.joint_vel[indices]
        limits = asset.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def init_motion_loader(
    env,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str | None,
    delay_reset_env_ratio: float,
    max_delay_steps: int,
) -> None:
    del env_ids
    MotionResetManager.initialize(env, motion_dir, recovery_dir)
    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay > 0 and max_delay_steps > 0:
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        mask[torch.randperm(env.num_envs, device=env.device)[:num_delay]] = True
        env.termination_manager = DelayedTerminationManager(env.termination_manager, mask, max_delay_steps)
        print(f"[AMP] delayed reset: {num_delay}/{env.num_envs} envs, {max_delay_steps} steps")


def reset_from_motion_data(
    env,
    env_ids: torch.Tensor,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    MotionResetManager.get(motion_dir).reset(env, env_ids, motion_dir, asset_cfg)


def initialize_encoder_bias(
    env,
    env_ids: torch.Tensor | None,
    bias_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    del env_ids
    asset: Articulation = env.scene[asset_cfg.name]
    bias = torch.empty((env.num_envs, asset.num_joints), device=env.device)
    env._amp_mjlab_encoder_bias = bias.uniform_(*bias_range)

