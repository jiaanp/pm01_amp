from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from amp_mjlab_pm01 import MOTION_ROOT
from amp_mjlab_pm01.assets.pm01 import PM01_24DOF_CFG, PM01_ACTION_SCALE, PM01_JOINT_NAMES
from amp_mjlab_pm01.tasks.amp_loco import mdp


PM01_AMP_BODY_NAMES = (
    "LINK_BASE",
    "LINK_HIP_ROLL_L", "LINK_KNEE_PITCH_L", "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R", "LINK_KNEE_PITCH_R", "LINK_ANKLE_ROLL_R",
    "LINK_SHOULDER_ROLL_L", "LINK_ELBOW_PITCH_L", "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_ROLL_R", "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R",
)
JOINT_CFG = SceneEntityCfg("robot", joint_names=list(PM01_JOINT_NAMES), preserve_order=True)
ANCHOR_CFG = SceneEntityCfg("robot", body_names=["LINK_TORSO_YAW"], preserve_order=True)
AMP_BODY_CFG = SceneEntityCfg("robot", body_names=list(PM01_AMP_BODY_NAMES), preserve_order=True)


@configclass
class PM01AmpSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = PM01_24DOF_CFG
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.005,
        history_length=4,
        track_air_time=True,
    )
    self_contacts = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.005,
        history_length=4,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/(LINK_BASE|LINK_HIP_PITCH_L|LINK_HIP_ROLL_L|LINK_HIP_YAW_L|LINK_KNEE_PITCH_L|LINK_ANKLE_PITCH_L|LINK_ANKLE_ROLL_L|LINK_HIP_PITCH_R|LINK_HIP_ROLL_R|LINK_HIP_YAW_R|LINK_KNEE_PITCH_R|LINK_ANKLE_PITCH_R|LINK_ANKLE_ROLL_R|LINK_TORSO_YAW|LINK_SHOULDER_PITCH_L|LINK_SHOULDER_ROLL_L|LINK_SHOULDER_YAW_L|LINK_ELBOW_PITCH_L|LINK_ELBOW_YAW_L|LINK_SHOULDER_PITCH_R|LINK_SHOULDER_ROLL_R|LINK_SHOULDER_YAW_R|LINK_ELBOW_PITCH_R|LINK_ELBOW_YAW_R|LINK_HEAD_YAW)"],
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


@configclass
class CommandsCfg:
    twist = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.25,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.5, 3.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-math.pi / 2, math.pi / 2),
            heading=(-math.pi / 2, math.pi / 2),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(PM01_JOINT_NAMES),
        scale=PM01_ACTION_SCALE,
        use_default_offset=True,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        frame = ObsTerm(
            func=mdp.policy_frame,
            params={"command_name": "twist", "asset_cfg": JOINT_CFG, "noisy": True},
            history_length=4,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        frame = ObsTerm(
            func=mdp.critic_frame,
            params={
                "command_name": "twist",
                "asset_cfg": JOINT_CFG,
                "anchor_cfg": ANCHOR_CFG,
                "body_cfg": AMP_BODY_CFG,
            },
            history_length=4,
            flatten_history_dim=True,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        frame = ObsTerm(
            func=mdp.amp_frame,
            params={"anchor_cfg": ANCHOR_CFG, "body_cfg": AMP_BODY_CFG},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class EventsCfg:
    init_motion_loader = EventTerm(
        func=mdp.init_stratified_recovery_loader,
        mode="startup",
        params={
            "motion_dir": str(MOTION_ROOT / "WalkandRun"),
            "recovery_dir": str(MOTION_ROOT / "Recovery"),
            "delay_reset_env_ratio": 0.7,
            "max_delay_steps": 250,
            "recovery_hard_ratio": 0.6,
            "recovery_semi_ratio": 0.2,
            "recovery_other_ratio": 0.2,
            "stable_height": 0.68,
            "stable_tilt_deg": 15.0,
            "stable_steps": 25,
        },
    )
    encoder_bias = EventTerm(
        func=mdp.initialize_encoder_bias,
        mode="startup",
        params={"bias_range": (-0.015, 0.015), "asset_cfg": SceneEntityCfg("robot")},
    )
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True
            ),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_TORSO_YAW"]),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.025, 0.025), "z": (-0.03, 0.03)},
        },
    )
    reset_from_motion = EventTerm(
        func=mdp.reset_from_stratified_motion_data,
        mode="reset",
        params={
            "motion_dir": str(MOTION_ROOT / "WalkandRun"),
            "asset_cfg": JOINT_CFG,
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={
            "velocity_range": {
                "x": (-1.0, 1.0), "y": (-0.5, 0.5), "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78),
            }
        },
    )


@configclass
class RewardsCfg:
    track_anchor_linear_velocity = RewTerm(
        func=mdp.track_anchor_linear_velocity,
        weight=1.0,
        params={
            "command_name": "twist", "std": 1.0, "mask_delay": True,
            "delay_env_rew_ratio": 0.0, "anchor_cfg": ANCHOR_CFG,
        },
    )
    track_anchor_angular_velocity = RewTerm(
        func=mdp.track_anchor_angular_velocity,
        weight=1.0,
        params={
            "command_name": "twist", "std": math.pi, "mask_delay": True,
            "delay_env_rew_ratio": 0.0, "anchor_cfg": ANCHOR_CFG,
        },
    )
    track_root_height = RewTerm(
        func=mdp.track_pm01_root_height,
        weight=1.0,
        params={"std": 0.3, "target_height": 0.76, "mask_delay": True, "delay_env_rew_ratio": 3.5},
    )
    body_ang_vel_xy_l2 = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=0.5,
        params={
            "std": math.pi, "mask_delay": True, "delay_env_rew_ratio": 0.0,
            "body_cfg": SceneEntityCfg("robot", body_names=["LINK_BASE"]),
        },
    )
    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7, params={"asset_cfg": JOINT_CFG})
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": JOINT_CFG})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    foot_slip = RewTerm(
        func=mdp.feet_slip,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True
            ),
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"], preserve_order=True
            ),
        },
    )
    self_collisions = RewTerm(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": "self_contacts", "force_threshold": 10.0},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)})
    bad_base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.5})


@configclass
class CurriculumCfg:
    command_vel = CurrTerm(
        func=mdp.commands_vel,
        params={
            "command_name": "twist",
            "velocity_stages": [
                {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
                {"step": 5000 * 24, "lin_vel_x": (-1.0, 2.0), "lin_vel_y": (-1.0, 1.0)},
            ],
        },
    )


@configclass
class PM01AmpFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: PM01AmpSceneCfg = PM01AmpSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (3.0, 3.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class PM01AmpFlatEnvCfg_PLAY(PM01AmpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = 1.0e9
        self.observations.policy.frame.params["noisy"] = False
        self.events.push_robot = None
        self.curriculum.command_vel = None
        # Keep playback on the original uniform Recovery sampler; the stratified curriculum is training-only.
        self.events.init_motion_loader.func = mdp.init_motion_loader
        self.events.init_motion_loader.params = {
            "motion_dir": str(MOTION_ROOT / "WalkandRun"),
            "recovery_dir": str(MOTION_ROOT / "Recovery"),
            "delay_reset_env_ratio": 1.0,
            "max_delay_steps": 250,
        }
        self.events.reset_from_motion.func = mdp.reset_from_motion_data

