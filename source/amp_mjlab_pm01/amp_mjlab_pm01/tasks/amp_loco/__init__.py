import gymnasium as gym


def _register(task_id: str, env_cfg_entry_point: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg_entry_point,
            "rsl_rl_cfg_entry_point": "amp_mjlab_pm01.tasks.amp_loco.agents.amp_ppo_cfg:PM01AmpPPORunnerCfg",
        },
    )


_register("AMP-MJLAB-PM01-Phase1-v0", "amp_mjlab_pm01.tasks.amp_loco.phase1_env_cfg:PM01AmpPhase1EnvCfg")
_register("AMP-MJLAB-PM01-Phase2-v0", "amp_mjlab_pm01.tasks.amp_loco.env_cfg:PM01AmpFlatEnvCfg")
_register("AMP-MJLAB-PM01-Phase2-Play-v0", "amp_mjlab_pm01.tasks.amp_loco.env_cfg:PM01AmpFlatEnvCfg_PLAY")
