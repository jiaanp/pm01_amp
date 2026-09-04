from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

from .events import _MotionFrames, _load_motion_dir
from .rewards import _delay_only
from .terminations import DelayedTerminationManager


# Reset metadata used by the diagnostic logger.
SUPINE = 0
PRONE = 1
RIGHT_SIDE = 2
LEFT_SIDE = 3
SEMI_FALL = 4
OTHER_RECOVERY = 5
HARD_FALL_NAMES = ("supine", "prone", "right_side", "left_side")

# Velocity modes are tracked independently from the pose class so that strict
# recovery metrics can distinguish dynamic reference resets from motionless starts.
DYNAMIC_RESET = 0
STATIC_RESET = 1
LOW_SPEED_RESET = 2


class StratifiedRecoveryResetManager:
    """PM01 recovery sampler layered on top of the AMP_mjlab reset semantics."""

    _instances: dict[str, "StratifiedRecoveryResetManager"] = {}

    def __init__(
        self,
        walk: _MotionFrames,
        recovery: _MotionFrames,
        sample_ratios: tuple[float, float, float],
        hard_static_ratio: float,
        hard_low_speed_ratio: float,
        low_root_lin_vel: float,
        low_root_ang_vel: float,
        low_joint_vel: float,
        hard_default_joint_pos: bool = False,
        hard_orientation_ids: tuple[int, ...] = (SUPINE, PRONE, RIGHT_SIDE, LEFT_SIDE),
    ) -> None:
        self.walk = walk
        self.recovery = recovery
        self.sample_ratios = sample_ratios
        self.hard_velocity_ratios = (
            1.0 - hard_static_ratio - hard_low_speed_ratio,
            hard_static_ratio,
            hard_low_speed_ratio,
        )
        self.low_root_lin_vel = low_root_lin_vel
        self.low_root_ang_vel = low_root_ang_vel
        self.low_joint_vel = low_joint_vel
        self.hard_default_joint_pos = hard_default_joint_pos
        if not hard_orientation_ids or any(index < SUPINE or index > LEFT_SIDE for index in hard_orientation_ids):
            raise ValueError(f"Invalid hard orientation ids: {hard_orientation_ids}")
        self.hard_orientation_ids = hard_orientation_ids
        self.recovery_bins = self._build_recovery_bins(recovery)

    @classmethod
    def initialize(
        cls,
        env,
        motion_dir: str,
        recovery_dir: str,
        sample_ratios: tuple[float, float, float],
        hard_static_ratio: float = 0.0,
        hard_low_speed_ratio: float = 0.0,
        low_root_lin_vel: float = 0.05,
        low_root_ang_vel: float = 0.10,
        low_joint_vel: float = 0.10,
        hard_default_joint_pos: bool = False,
        hard_orientation_ids: tuple[int, ...] = (SUPINE, PRONE, RIGHT_SIDE, LEFT_SIDE),
    ) -> None:
        asset: Articulation = env.scene["robot"]
        joint_names = list(asset.joint_names)
        cls._instances[motion_dir] = cls(
            _load_motion_dir(motion_dir, env.device, joint_names),
            _load_motion_dir(recovery_dir, env.device, joint_names),
            sample_ratios,
            hard_static_ratio,
            hard_low_speed_ratio,
            low_root_lin_vel,
            low_root_ang_vel,
            low_joint_vel,
            hard_default_joint_pos,
            hard_orientation_ids,
        )

    @classmethod
    def get(cls, motion_dir: str) -> "StratifiedRecoveryResetManager":
        return cls._instances[motion_dir]

    @staticmethod
    def _world_up_in_body(quat_wxyz: torch.Tensor) -> torch.Tensor:
        w, x, y, z = quat_wxyz.unbind(dim=-1)
        return torch.stack(
            (
                2.0 * (x * z - w * y),
                2.0 * (y * z + w * x),
                1.0 - 2.0 * (x * x + y * y),
            ),
            dim=-1,
        )

    @classmethod
    def _build_recovery_bins(cls, frames: _MotionFrames) -> tuple[torch.Tensor, ...]:
        up_b = cls._world_up_in_body(frames.root_quat)
        hard = (frames.root_pos[:, 2] < 0.25) & (
            up_b[:, 2] < math.cos(math.radians(75.0))
        )
        fallen = (frames.root_pos[:, 2] < 0.5) | (
            up_b[:, 2] < math.cos(math.radians(70.0))
        )
        pitch_dominant = torch.abs(up_b[:, 0]) >= torch.abs(up_b[:, 1])
        masks = (
            hard & pitch_dominant & (up_b[:, 0] >= 0.0),
            hard & pitch_dominant & (up_b[:, 0] < 0.0),
            hard & ~pitch_dominant & (up_b[:, 1] >= 0.0),
            hard & ~pitch_dominant & (up_b[:, 1] < 0.0),
            fallen & ~hard,
            ~fallen,
        )
        bins = tuple(mask.nonzero(as_tuple=False).squeeze(-1) for mask in masks)
        counts = [int(indices.numel()) for indices in bins]
        if sum(counts[:4]) == 0:
            raise RuntimeError("Recovery dataset contains no strict hard-fall frames")
        print(
            "[AMP] recovery frame bins: "
            f"supine={counts[0]}, prone={counts[1]}, right={counts[2]}, left={counts[3]}, "
            f"semi={counts[4]}, other={counts[5]}"
        )
        return bins

    def _sample_recovery(
        self, count: int, device: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        group = torch.multinomial(
            torch.tensor(self.sample_ratios, dtype=torch.float32, device=device),
            count,
            replacement=True,
        )
        frame_indices = torch.empty(count, dtype=torch.long, device=device)
        reset_classes = torch.empty(count, dtype=torch.long, device=device)
        velocity_modes = torch.full((count,), DYNAMIC_RESET, dtype=torch.long, device=device)

        hard_rows = (group == 0).nonzero(as_tuple=False).squeeze(-1)
        if hard_rows.numel() > 0:
            # Equal counts per direction within each reset batch, with randomized row assignment.
            orientation_count = len(self.hard_orientation_ids)
            orientations = torch.arange(hard_rows.numel(), device=device) % orientation_count
            orientations = orientations[torch.randperm(hard_rows.numel(), device=device)]
            all_hard = torch.cat([self.recovery_bins[index] for index in self.hard_orientation_ids])
            for selection_index, orientation in enumerate(self.hard_orientation_ids):
                rows = hard_rows[orientations == selection_index]
                if rows.numel() == 0:
                    continue
                candidates = self.recovery_bins[orientation]
                if candidates.numel() == 0:
                    candidates = all_hard
                picks = torch.randint(candidates.numel(), (rows.numel(),), device=device)
                frame_indices[rows] = candidates[picks]
                reset_classes[rows] = orientation

            velocity_modes[hard_rows] = torch.multinomial(
                torch.tensor(self.hard_velocity_ratios, dtype=torch.float32, device=device),
                hard_rows.numel(),
                replacement=True,
            )

        for group_id, reset_class in ((1, SEMI_FALL), (2, OTHER_RECOVERY)):
            rows = (group == group_id).nonzero(as_tuple=False).squeeze(-1)
            if rows.numel() == 0:
                continue
            candidates = self.recovery_bins[reset_class]
            if candidates.numel() == 0:
                candidates = torch.arange(self.recovery.root_pos.shape[0], device=device)
            picks = torch.randint(candidates.numel(), (rows.numel(),), device=device)
            frame_indices[rows] = candidates[picks]
            reset_classes[rows] = reset_class
        return frame_indices, reset_classes, velocity_modes

    @staticmethod
    def _write(
        env,
        env_ids: torch.Tensor,
        frames: _MotionFrames,
        asset_cfg: SceneEntityCfg,
        indices: torch.Tensor | None = None,
        velocity_modes: torch.Tensor | None = None,
        low_root_lin_vel: float = 0.05,
        low_root_ang_vel: float = 0.10,
        low_joint_vel: float = 0.10,
        default_joint_pos: bool | torch.Tensor = False,
    ) -> None:
        asset: Articulation = env.scene[asset_cfg.name]
        if indices is None:
            indices = torch.randint(frames.root_pos.shape[0], (len(env_ids),), device=env.device)
        position = env.scene.env_origins[env_ids].clone()
        position[:, 2] += frames.root_pos[indices, 2]
        root_pose = torch.cat((position, frames.root_quat[indices]), dim=-1)
        root_velocity = torch.cat((frames.root_lin_vel[indices], frames.root_ang_vel[indices]), dim=-1)
        reference_joint_pos = frames.joint_pos[indices]
        if isinstance(default_joint_pos, torch.Tensor):
            joint_pos = torch.where(
                default_joint_pos[:, None], asset.data.default_joint_pos[env_ids], reference_joint_pos
            )
        else:
            joint_pos = asset.data.default_joint_pos[env_ids].clone() if default_joint_pos else reference_joint_pos
        joint_vel = frames.joint_vel[indices]
        if velocity_modes is not None:
            static = velocity_modes == STATIC_RESET
            low_speed = velocity_modes == LOW_SPEED_RESET
            root_velocity[static] = 0.0
            joint_vel[static] = 0.0
            if low_speed.any():
                count = int(low_speed.sum().item())
                root_velocity[low_speed, :3] = torch.empty((count, 3), device=env.device).uniform_(
                    -low_root_lin_vel, low_root_lin_vel
                )
                root_velocity[low_speed, 3:] = torch.empty((count, 3), device=env.device).uniform_(
                    -low_root_ang_vel, low_root_ang_vel
                )
                joint_vel[low_speed] = torch.empty(
                    (count, joint_vel.shape[1]), device=env.device
                ).uniform_(-low_joint_vel, low_joint_vel)
        limits = asset.data.soft_joint_pos_limits[env_ids]
        joint_pos = joint_pos.clamp(limits[..., 0], limits[..., 1])
        asset.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_link_velocity_to_sim(root_velocity, env_ids=env_ids)
        asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    def reset(self, env, env_ids: torch.Tensor, asset_cfg: SceneEntityCfg) -> None:
        delay_mask = getattr(env.termination_manager, "_delay_env_mask", None)
        if delay_mask is None:
            self._write(env, env_ids, self.walk, asset_cfg)
            return

        is_delay = delay_mask[env_ids]
        normal_ids = env_ids[~is_delay]
        delay_ids = env_ids[is_delay]
        reset_classes = torch.full((len(env_ids),), -1, dtype=torch.long, device=env.device)
        velocity_modes = torch.full_like(reset_classes, -1)
        if normal_ids.numel() > 0:
            self._write(env, normal_ids, self.walk, asset_cfg)
        if delay_ids.numel() > 0:
            indices, classes, modes = self._sample_recovery(len(delay_ids), env.device)
            self._write(
                env,
                delay_ids,
                self.recovery,
                asset_cfg,
                indices,
                modes,
                self.low_root_lin_vel,
                self.low_root_ang_vel,
                self.low_joint_vel,
                self.hard_default_joint_pos & (classes >= SUPINE) & (classes <= LEFT_SIDE),
            )
            reset_classes[is_delay] = classes
            velocity_modes[is_delay] = modes
        env.termination_manager.queue_recovery_reset_metadata(env_ids, reset_classes, velocity_modes)


class RecoveryMetricsDelayedTerminationManager(DelayedTerminationManager):
    """Delayed termination plus strict hard-fall success and constraint diagnostics."""

    def __init__(
        self,
        base,
        delay_env_mask: torch.Tensor,
        max_delay_steps: int,
        stable_height: float,
        stable_tilt_deg: float,
        stable_steps: int,
    ) -> None:
        super().__init__(base, delay_env_mask, max_delay_steps)
        count = self.num_envs
        self._recovery_class = torch.full((count,), -1, dtype=torch.long, device=self.device)
        self._pending_recovery_class = torch.full_like(self._recovery_class, -1)
        self._recovery_velocity_mode = torch.full_like(self._recovery_class, -1)
        self._pending_recovery_velocity_mode = torch.full_like(self._recovery_class, -1)
        self._stable_counter = torch.zeros(count, dtype=torch.long, device=self.device)
        self._getup_success = torch.zeros(count, dtype=torch.bool, device=self.device)
        self._time_to_stand = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._torque_saturation_sum = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._soft_limit_violation_sum = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._diagnostic_steps = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._first_target_jump = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._peak_target_error = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._peak_joint_speed = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._peak_torque_ratio = torch.zeros(count, dtype=torch.float32, device=self.device)
        self._stable_height = stable_height
        self._stable_cos_tilt = math.cos(math.radians(stable_tilt_deg))
        self._stable_steps = stable_steps

    def queue_recovery_reset_metadata(
        self,
        env_ids: torch.Tensor,
        reset_classes: torch.Tensor,
        velocity_modes: torch.Tensor | None = None,
    ) -> None:
        self._pending_recovery_class[env_ids] = reset_classes
        if velocity_modes is not None:
            self._pending_recovery_velocity_mode[env_ids] = velocity_modes

    def compute(self) -> torch.Tensor:
        dones = super().compute()
        asset: Articulation = self._env.scene["robot"]
        hard = (self._recovery_class >= SUPINE) & (self._recovery_class <= LEFT_SIDE)
        quat = asset.data.body_link_quat_w[:, 0]
        up_z = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square())
        linear_speed = torch.linalg.vector_norm(asset.data.body_link_lin_vel_w[:, 0], dim=-1)
        angular_speed = torch.linalg.vector_norm(asset.data.body_link_ang_vel_w[:, 0], dim=-1)
        stable = (
            hard
            & (asset.data.body_link_pos_w[:, 0, 2] >= self._stable_height)
            & (up_z >= self._stable_cos_tilt)
            & (linear_speed < 0.3)
            & (angular_speed < 0.5)
        )
        self._stable_counter = torch.where(stable, self._stable_counter + 1, 0)
        new_success = hard & ~self._getup_success & (self._stable_counter >= self._stable_steps)
        self._getup_success |= new_success
        self._time_to_stand[new_success] = (
            self._env.episode_length_buf[new_success].float() * self._env.step_dt
        )

        efforts = asset.data.applied_torque
        effort_limits = asset.data.joint_effort_limits
        target_error = torch.max(torch.abs(asset.data.joint_pos_target - asset.data.joint_pos), dim=1).values
        joint_speed = torch.max(torch.abs(asset.data.joint_vel), dim=1).values
        first_step = hard & (self._diagnostic_steps == 0)
        self._first_target_jump[first_step] = target_error[first_step]
        self._peak_target_error = torch.maximum(self._peak_target_error, target_error)
        self._peak_joint_speed = torch.maximum(self._peak_joint_speed, joint_speed)
        if efforts is not None and effort_limits is not None:
            saturated = efforts.abs() >= 0.98 * effort_limits.clamp_min(1.0e-6)
            self._torque_saturation_sum += saturated.float().mean(dim=1)
            torque_ratio = torch.max(
                efforts.abs() / effort_limits.clamp_min(1.0e-6), dim=1
            ).values
            self._peak_torque_ratio = torch.maximum(self._peak_torque_ratio, torque_ratio)
        limits = asset.data.soft_joint_pos_limits
        outside = (asset.data.joint_pos < limits[..., 0]) | (asset.data.joint_pos > limits[..., 1])
        self._soft_limit_violation_sum += outside.float().mean(dim=1)
        self._diagnostic_steps += 1.0
        return dones

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        old_classes = self._recovery_class[ids]
        old_modes = self._recovery_velocity_mode[ids]
        hard = (old_classes >= SUPINE) & (old_classes <= LEFT_SIDE)
        extras = super().reset(env_ids)

        if hard.any():
            success = self._getup_success[ids][hard]
            extras["Recovery/strict_hard_getup_success"] = success.float().mean().item()
            successful_times = self._time_to_stand[ids][hard][success]
            extras["Recovery/strict_hard_time_to_stand_s"] = (
                successful_times.mean().item() if successful_times.numel() > 0 else 0.0
            )
            steps = self._diagnostic_steps[ids][hard].clamp_min(1.0)
            extras["Recovery/strict_hard_torque_saturation_rate"] = (
                self._torque_saturation_sum[ids][hard] / steps
            ).mean().item()
            extras["Recovery/strict_hard_soft_limit_violation_rate"] = (
                self._soft_limit_violation_sum[ids][hard] / steps
            ).mean().item()
            extras["Recovery/strict_hard_first_target_jump_rad"] = self._first_target_jump[ids][hard].mean().item()
            extras["Recovery/strict_hard_peak_target_error_rad"] = self._peak_target_error[ids][hard].mean().item()
            extras["Recovery/strict_hard_peak_joint_speed_rad_s"] = self._peak_joint_speed[ids][hard].mean().item()
            extras["Recovery/strict_hard_peak_torque_ratio"] = self._peak_torque_ratio[ids][hard].mean().item()
            for velocity_mode, name in (
                (DYNAMIC_RESET, "dynamic"),
                (STATIC_RESET, "static"),
                (LOW_SPEED_RESET, "low_speed"),
            ):
                mode_mask = hard & (old_modes == velocity_mode)
                if mode_mask.any():
                    extras[f"Recovery/{name}_hard_success"] = (
                        self._getup_success[ids][mode_mask].float().mean().item()
                    )
            for recovery_class, name in enumerate(HARD_FALL_NAMES):
                orientation = old_classes == recovery_class
                if orientation.any():
                    extras[f"Recovery/{name}_success"] = (
                        self._getup_success[ids][orientation].float().mean().item()
                    )

        new_classes = self._pending_recovery_class[ids]
        new_modes = self._pending_recovery_velocity_mode[ids]
        delayed = new_classes >= SUPINE
        if delayed.any():
            extras["Recovery_Reset/strict_hard_fraction"] = (
                (new_classes[delayed] <= LEFT_SIDE).float().mean().item()
            )
            extras["Recovery_Reset/semi_fall_fraction"] = (
                (new_classes[delayed] == SEMI_FALL).float().mean().item()
            )
            hard_new = (new_classes >= SUPINE) & (new_classes <= LEFT_SIDE)
            if hard_new.any():
                extras["Recovery_Reset/static_hard_fraction"] = (
                    (new_modes[hard_new] == STATIC_RESET).float().mean().item()
                )
                extras["Recovery_Reset/low_speed_hard_fraction"] = (
                    (new_modes[hard_new] == LOW_SPEED_RESET).float().mean().item()
                )
        self._recovery_class[ids] = new_classes
        self._pending_recovery_class[ids] = -1
        self._recovery_velocity_mode[ids] = new_modes
        self._pending_recovery_velocity_mode[ids] = -1
        self._stable_counter[ids] = 0
        self._getup_success[ids] = False
        self._time_to_stand[ids] = 0.0
        self._torque_saturation_sum[ids] = 0.0
        self._soft_limit_violation_sum[ids] = 0.0
        self._diagnostic_steps[ids] = 0.0
        self._first_target_jump[ids] = 0.0
        self._peak_target_error[ids] = 0.0
        self._peak_joint_speed[ids] = 0.0
        self._peak_torque_ratio[ids] = 0.0
        return extras


def init_stratified_recovery_loader(
    env,
    env_ids: torch.Tensor | None,
    motion_dir: str,
    recovery_dir: str,
    delay_reset_env_ratio: float,
    max_delay_steps: int,
    recovery_hard_ratio: float,
    recovery_semi_ratio: float,
    recovery_other_ratio: float,
    stable_height: float,
    stable_tilt_deg: float,
    stable_steps: int,
    hard_static_ratio: float = 0.0,
    hard_low_speed_ratio: float = 0.0,
    low_root_lin_vel: float = 0.05,
    low_root_ang_vel: float = 0.10,
    low_joint_vel: float = 0.10,
    hard_default_joint_pos: bool = False,
    hard_orientation_ids: tuple[int, ...] = (SUPINE, PRONE, RIGHT_SIDE, LEFT_SIDE),
) -> None:
    del env_ids
    ratios = (recovery_hard_ratio, recovery_semi_ratio, recovery_other_ratio)
    if any(ratio < 0.0 for ratio in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1.0e-6):
        raise ValueError(f"Recovery sample ratios must be non-negative and sum to one: {ratios}")
    if hard_static_ratio < 0.0 or hard_low_speed_ratio < 0.0 or hard_static_ratio + hard_low_speed_ratio > 1.0:
        raise ValueError(
            "Hard reset velocity ratios must be non-negative and sum to at most one: "
            f"static={hard_static_ratio}, low_speed={hard_low_speed_ratio}"
        )
    StratifiedRecoveryResetManager.initialize(
        env,
        motion_dir,
        recovery_dir,
        ratios,
        hard_static_ratio,
        hard_low_speed_ratio,
        low_root_lin_vel,
        low_root_ang_vel,
        low_joint_vel,
        hard_default_joint_pos,
        hard_orientation_ids,
    )
    num_delay = int(env.num_envs * delay_reset_env_ratio)
    if num_delay > 0 and max_delay_steps > 0:
        mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        mask[torch.randperm(env.num_envs, device=env.device)[:num_delay]] = True
        env.termination_manager = RecoveryMetricsDelayedTerminationManager(
            env.termination_manager,
            mask,
            max_delay_steps,
            stable_height,
            stable_tilt_deg,
            stable_steps,
        )
        print(f"[AMP] delayed recovery curriculum: {num_delay}/{env.num_envs} envs, {max_delay_steps} steps")


def reset_from_stratified_motion_data(
    env,
    env_ids: torch.Tensor,
    motion_dir: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    StratifiedRecoveryResetManager.get(motion_dir).reset(env, env_ids, asset_cfg)


def track_pm01_root_height(
    env,
    std: float,
    target_height: float,
    mask_delay: bool,
    delay_env_rew_ratio: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    error = torch.square(target_height - asset.data.body_link_pos_w[:, 0, 2])
    return _delay_only(env, torch.exp(-error / std**2), mask_delay, delay_env_rew_ratio)
