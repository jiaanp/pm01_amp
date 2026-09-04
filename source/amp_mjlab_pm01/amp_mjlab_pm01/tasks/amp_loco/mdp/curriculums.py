from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from isaaclab.envs import ManagerBasedRLEnv


def commands_vel(env: ManagerBasedRLEnv, env_ids: torch.Tensor, command_name: str, velocity_stages: list[dict]):
    del env_ids
    cfg = env.command_manager.get_term(command_name).cfg
    for stage in velocity_stages:
        if env.common_step_counter > stage["step"]:
            for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
                if stage.get(key) is not None:
                    setattr(cfg.ranges, key, stage[key])
    return {}

