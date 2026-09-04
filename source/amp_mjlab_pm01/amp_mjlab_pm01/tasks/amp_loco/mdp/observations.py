from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _encoder_bias(env: ManagerBasedRLEnv, asset: Articulation, joint_ids) -> torch.Tensor:
    bias = getattr(env, "_amp_mjlab_encoder_bias", None)
    if bias is None:
        return torch.zeros_like(asset.data.joint_pos[:, joint_ids])
    return bias[:, joint_ids]


def _uniform_noise(value: torch.Tensor, magnitude: float, enabled: bool) -> torch.Tensor:
    if not enabled:
        return value
    return value + torch.empty_like(value).uniform_(-magnitude, magnitude)


def _policy_command(env: ManagerBasedRLEnv, command_name: str, zero_on_recovery: bool) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    if not zero_on_recovery:
        return command
    # Observation terms are evaluated once for shape discovery before the
    # termination manager is constructed. In that phase there is no recovery
    # mask yet; normal runtime calls see the wrapped termination manager.
    termination_manager = getattr(env, "termination_manager", None)
    recovery_mask = getattr(termination_manager, "_delay_env_mask", None)
    if not isinstance(recovery_mask, torch.Tensor):
        return command
    return torch.where(recovery_mask[:, None], torch.zeros_like(command), command)


def robot_body_pos_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[anchor_cfg.name]
    anchor_pos = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
    anchor_quat = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    body_pos = asset.data.body_link_pos_w[:, body_cfg.body_ids]
    body_quat = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    count = body_pos.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos[:, None].expand(-1, count, -1),
        anchor_quat[:, None].expand(-1, count, -1),
        body_pos,
        body_quat,
    )
    return pos_b.reshape(env.num_envs, -1)


def robot_body_ori_b(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[anchor_cfg.name]
    anchor_pos = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]
    anchor_quat = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]
    body_pos = asset.data.body_link_pos_w[:, body_cfg.body_ids]
    body_quat = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    count = body_pos.shape[1]
    _, quat_b = subtract_frame_transforms(
        anchor_pos[:, None].expand(-1, count, -1),
        anchor_quat[:, None].expand(-1, count, -1),
        body_pos,
        body_quat,
    )
    matrix = matrix_from_quat(quat_b)
    return matrix[..., :2].reshape(env.num_envs, -1)


def robot_body_lin_vel_b(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[body_cfg.name]
    velocity = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]
    quaternion = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    count = velocity.shape[1]
    local = quat_apply_inverse(quaternion.reshape(-1, 4), velocity.reshape(-1, 3))
    return local.reshape(env.num_envs, count * 3)


def robot_body_ang_vel_b(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: Articulation = env.scene[body_cfg.name]
    velocity = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]
    quaternion = asset.data.body_link_quat_w[:, body_cfg.body_ids]
    count = velocity.shape[1]
    local = quat_apply_inverse(quaternion.reshape(-1, 4), velocity.reshape(-1, 3))
    return local.reshape(env.num_envs, count * 3)


def policy_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    noisy: bool = True,
    zero_command_on_recovery: bool = False,
) -> torch.Tensor:
    """One complete policy frame; history on this term preserves time ordering."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    joint_pos = asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    joint_pos = joint_pos + _encoder_bias(env, asset, joint_ids)
    joint_vel = asset.data.joint_vel[:, joint_ids] - asset.data.default_joint_vel[:, joint_ids]
    return torch.cat(
        (
            _uniform_noise(asset.data.root_ang_vel_b, 0.2, noisy),
            _uniform_noise(asset.data.projected_gravity_b, 0.05, noisy),
            _policy_command(env, command_name, zero_command_on_recovery),
            _uniform_noise(joint_pos, 0.01, noisy),
            _uniform_noise(joint_vel, 0.5, noisy),
            env.action_manager.action,
        ),
        dim=-1,
    )


def recovery_start_phase(env: ManagerBasedRLEnv, action_name: str = "joint_pos") -> torch.Tensor:
    """A deployment-visible phase for a strict-static recovery startup.

    The action term owns the clock so the policy sees exactly the phase that
    governs its applied joint targets.  Non-recovery environments use 1.0.
    """
    action_manager = getattr(env, "action_manager", None)
    term = action_manager.get_term(action_name) if action_manager is not None else None
    phase = getattr(term, "recovery_phase", None)
    if not isinstance(phase, torch.Tensor):
        return torch.ones((env.num_envs, 1), device=env.device)
    return phase.unsqueeze(-1)


def critic_frame(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    anchor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    zero_command_on_recovery: bool = False,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    actor = policy_frame(
        env,
        command_name,
        asset_cfg,
        noisy=False,
        zero_command_on_recovery=zero_command_on_recovery,
    )
    return torch.cat(
        (
            actor,
            asset.data.root_lin_vel_b,
            robot_body_pos_b(env, anchor_cfg, body_cfg),
            robot_body_ori_b(env, anchor_cfg, body_cfg),
        ),
        dim=-1,
    )


def amp_frame(
    env: ManagerBasedRLEnv,
    anchor_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return torch.cat(
        (
            robot_body_pos_b(env, anchor_cfg, body_cfg),
            robot_body_ori_b(env, anchor_cfg, body_cfg),
            robot_body_lin_vel_b(env, body_cfg),
            robot_body_ang_vel_b(env, body_cfg),
        ),
        dim=-1,
    )
