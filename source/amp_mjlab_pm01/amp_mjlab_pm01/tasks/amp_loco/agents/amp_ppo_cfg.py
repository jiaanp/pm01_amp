from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from amp_mjlab_pm01 import MOTION_ROOT
from amp_mjlab_pm01.tasks.amp_loco.env_cfg import PM01_AMP_BODY_NAMES


@configclass
class PM01AmpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    class_name = "AmpOnPolicyRunner"
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 100001
    save_interval = 100
    experiment_name = "pm01_amp_mjlab_locomotion"
    logger = "tensorboard"
    clip_actions = None
    empirical_normalization = True
    obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="AMPPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    amp_reward_coef = 0.1
    amp_motion_files = str(MOTION_ROOT)
    amp_num_preload_transitions = 200000
    amp_task_reward_lerp = 0.75
    amp_discr_hidden_dims = [1024, 512, 256]
    min_normalized_std = [0.05] * 24
    amp_body_names = PM01_AMP_BODY_NAMES
    amp_anchor_name = "LINK_TORSO_YAW"

