"""Frozen Phase-1 environment used for the original 0 -> 100000 run."""

from isaaclab.utils import configclass

from . import mdp
from .env_cfg import PM01AmpFlatEnvCfg


@configclass
class PM01AmpPhase1EnvCfg(PM01AmpFlatEnvCfg):
    """Original uniform reset curriculum, frozen from the model_100000 snapshot."""

    def __post_init__(self):
        super().__post_init__()
        self.events.init_motion_loader.func = mdp.init_motion_loader
        self.events.init_motion_loader.params = {
            "motion_dir": str(self.events.init_motion_loader.params["motion_dir"]),
            "recovery_dir": str(self.events.init_motion_loader.params["recovery_dir"]),
            "delay_reset_env_ratio": 0.4,
            "max_delay_steps": 250,
        }
        self.events.reset_from_motion.func = mdp.reset_from_motion_data
        self.rewards.track_root_height.func = mdp.track_root_height
        self.rewards.track_root_height.params = {
            "std": 0.3,
            "mask_delay": True,
            "delay_env_rew_ratio": 3.5,
        }
