from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _active_delay_mask(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    mask = getattr(env.termination_manager, "_delay_env_mask", None)
    counters = getattr(env.termination_manager, "_delay_counters", None)
    if isinstance(mask, torch.Tensor) and isinstance(counters, torch.Tensor):
        return mask & (counters > 0)
    return None


def _scale_delay(env, reward, enabled: bool, ratio: float):
    if not enabled:
        return reward
    mask = _active_delay_mask(env)
    if mask is None:
        return reward
    return torch.where(mask, reward * ratio, reward)


def _delay_only(env, reward, enabled: bool, ratio: float):
    if not enabled:
        return torch.zeros_like(reward)
    mask = _active_delay_mask(env)
    if mask is None:
        return torch.zeros_like(reward)
    return torch.where(mask, reward * ratio, torch.zeros_like(reward))


def _recovery_only(env, value: torch.Tensor) -> torch.Tensor:
    """Keep a reward/cost only on the dedicated recovery environments."""
    mask = getattr(env.termination_manager, "_delay_env_mask", None)
    if not isinstance(mask, torch.Tensor):
        return torch.zeros_like(value)
    return torch.where(mask, value, torch.zeros_like(value))


def recovery_joint_target_rate_l2(
    env: ManagerBasedRLEnv,
    initial_steps: int,
    initial_multiplier: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Physical q-target change cost with a current-q reference on the first step.

    Raw action zero maps to the default standing pose and is not a safe hold
    command for a fallen robot. This term therefore regularizes the command the
    actuator actually receives instead of blindly shrinking the network output.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    target = asset.data.joint_pos_target[:, joint_ids]
    current = asset.data.joint_pos[:, joint_ids]
    previous = getattr(env, "_pm01_previous_joint_target", None)
    if not isinstance(previous, torch.Tensor) or previous.shape != target.shape:
        previous = current
    first = env.episode_length_buf <= 1
    reference = torch.where(first[:, None], current, previous)
    cost = torch.mean(torch.square(target - reference), dim=1)
    early = env.episode_length_buf < initial_steps
    cost = torch.where(early, cost * initial_multiplier, cost)
    env._pm01_previous_joint_target = target.detach().clone()
    return _recovery_only(env, cost)


def recovery_normalized_torque_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    ratio = asset.data.applied_torque[:, joint_ids] / asset.data.joint_effort_limits[:, joint_ids].clamp_min(1.0e-6)
    return _recovery_only(env, torch.mean(torch.square(ratio), dim=1))


def recovery_joint_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = asset.data.joint_vel[:, asset_cfg.joint_ids]
    return _recovery_only(env, torch.mean(torch.square(velocity), dim=1))


def recovery_stable_upright(
    env: ManagerBasedRLEnv,
    target_height: float,
    height_std: float,
    tilt_std: float,
    linear_velocity_std: float,
    angular_velocity_std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward an upright, quiet final state instead of height alone."""
    asset: Articulation = env.scene[asset_cfg.name]
    height_error = torch.square(target_height - asset.data.body_link_pos_w[:, 0, 2])
    up_error = torch.square(1.0 + asset.data.projected_gravity_b[:, 2])
    linear_speed_sq = torch.sum(torch.square(asset.data.root_lin_vel_b), dim=1)
    angular_speed_sq = torch.sum(torch.square(asset.data.root_ang_vel_b), dim=1)
    reward = torch.exp(
        -height_error / height_std**2
        -up_error / tilt_std**2
        -linear_speed_sq / linear_velocity_std**2
        -angular_speed_sq / angular_velocity_std**2
    )
    return _recovery_only(env, reward)


def track_anchor_linear_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    mask_delay: bool,
    delay_env_rew_ratio: float,
    anchor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
    quat = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    command_w = quat_apply(yaw_quat(quat), command_b)
    velocity = asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0]]
    reward = torch.exp(-torch.sum(torch.square(command_w - velocity), dim=1) / std**2)
    return _scale_delay(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    mask_delay: bool,
    delay_env_rew_ratio: float,
    anchor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[anchor_cfg.name]
    command = env.command_manager.get_command(command_name)
    angular_w = asset.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0]]
    yaw_error = torch.square(command[:, 2] - angular_w[:, 2])
    quat = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    angular_b = quat_apply_inverse(quat, angular_w)
    error = yaw_error + torch.sum(torch.square(angular_b[:, :2]), dim=-1)
    reward = torch.exp(-error / std**2)
    return _scale_delay(env, reward, mask_delay, delay_env_rew_ratio)


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    std: float,
    mask_delay: bool,
    delay_env_rew_ratio: float,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[body_cfg.name]
    angular_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids[0]]
    quat = asset.data.body_link_quat_w[:, body_cfg.body_ids[0]]
    angular_b = quat_apply_inverse(quat, angular_w)
    reward = torch.exp(-torch.sum(torch.square(angular_b[:, :2]), dim=-1) / std**2)
    return _scale_delay(env, reward, mask_delay, delay_env_rew_ratio)


def track_root_height(
    env: ManagerBasedRLEnv,
    std: float,
    mask_delay: bool,
    delay_env_rew_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(asset.data.default_root_state[:, 2] - asset.data.body_link_pos_w[:, 0, 2])
    return _delay_only(env, torch.exp(-error / std**2), mask_delay, delay_env_rew_ratio)


def feet_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)
    active = (torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold).float()
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    in_contact = (torch.norm(forces, dim=-1) > 1.0).float()
    speed_sq = torch.sum(torch.square(asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1)
    return torch.sum(speed_sq * in_contact, dim=1) * active


def self_collision_cost(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    force_threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    forces = sensor.data.force_matrix_w_history
    if forces is None:
        forces = sensor.data.net_forces_w_history.unsqueeze(3)
    hit = (torch.norm(forces, dim=-1) > force_threshold).any(dim=(2, 3))
    return hit.sum(dim=1).float()
