import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from amp_mjlab_pm01 import ASSET_ROOT


PM01_JOINT_NAMES = (
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_WAIST_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R", "J23_HEAD_YAW",
)

PM01_DEFAULT_JOINT_POS = {
    "J00_HIP_PITCH_L": -0.06, "J01_HIP_ROLL_L": 0.0, "J02_HIP_YAW_L": 0.0,
    "J03_KNEE_PITCH_L": 0.12, "J04_ANKLE_PITCH_L": -0.06, "J05_ANKLE_ROLL_L": 0.0,
    "J06_HIP_PITCH_R": -0.06, "J07_HIP_ROLL_R": 0.0, "J08_HIP_YAW_R": 0.0,
    "J09_KNEE_PITCH_R": 0.12, "J10_ANKLE_PITCH_R": -0.06, "J11_ANKLE_ROLL_R": 0.0,
    "J12_WAIST_YAW": 0.0,
    "J13_SHOULDER_PITCH_L": 0.0, "J14_SHOULDER_ROLL_L": 0.15,
    "J15_SHOULDER_YAW_L": 0.0, "J16_ELBOW_PITCH_L": -0.25, "J17_ELBOW_YAW_L": 0.0,
    "J18_SHOULDER_PITCH_R": 0.0, "J19_SHOULDER_ROLL_R": -0.15,
    "J20_SHOULDER_YAW_R": 0.0, "J21_ELBOW_PITCH_R": -0.25, "J22_ELBOW_YAW_R": 0.0,
    "J23_HEAD_YAW": 0.0,
}

PM01_ACTION_SCALE = {
    ".*HIP_PITCH.*": 0.25 * 164.0 / 110.0,
    ".*HIP_ROLL.*": 0.25 * 164.0 / 70.0,
    ".*HIP_YAW.*": 0.25 * 61.0 / 70.0,
    ".*KNEE_PITCH.*": 0.25 * 164.0 / 110.0,
    ".*ANKLE_(PITCH|ROLL).*": 0.25 * 54.9 / 30.0,
    ".*SHOULDER.*": 0.25 * 61.0 / 50.0,
    ".*ELBOW.*": 0.25 * 61.0 / 50.0,
    ".*WAIST_YAW.*": 0.25 * 61.0 / 50.0,
    ".*HEAD_YAW.*": 0.25 * 52.0 / 50.0,
}

PM01_24DOF_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ASSET_ROOT / "pm01" / "urdf" / "serial_pm01.urdf"),
        activate_contact_sensors=True,
        fix_base=False,
        replace_cylinders_with_capsules=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos=PM01_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness={
                ".*HIP_PITCH.*": 110.0, ".*HIP_ROLL.*": 70.0, ".*HIP_YAW.*": 70.0,
                ".*KNEE_PITCH.*": 110.0, ".*ANKLE_(PITCH|ROLL).*": 30.0,
                ".*SHOULDER.*": 50.0, ".*ELBOW.*": 50.0,
                ".*WAIST_YAW.*": 50.0, ".*HEAD_YAW.*": 50.0,
            },
            damping={
                ".*HIP_PITCH.*": 5.0, ".*HIP_ROLL.*": 3.0, ".*HIP_YAW.*": 3.0,
                ".*KNEE_PITCH.*": 5.0, ".*ANKLE_(PITCH|ROLL).*": 0.3,
                ".*SHOULDER.*": 0.3, ".*ELBOW.*": 0.3,
                ".*WAIST_YAW.*": 3.0, ".*HEAD_YAW.*": 0.3,
            },
            effort_limit={
                ".*HIP_PITCH.*": 164.0, ".*HIP_ROLL.*": 164.0, ".*HIP_YAW.*": 61.0,
                ".*KNEE_PITCH.*": 164.0, ".*ANKLE_(PITCH|ROLL).*": 54.9,
                ".*SHOULDER.*": 61.0, ".*ELBOW.*": 61.0,
                ".*WAIST_YAW.*": 61.0, ".*HEAD_YAW.*": 52.0,
            },
            effort_limit_sim={
                ".*HIP_PITCH.*": 164.0, ".*HIP_ROLL.*": 164.0, ".*HIP_YAW.*": 61.0,
                ".*KNEE_PITCH.*": 164.0, ".*ANKLE_(PITCH|ROLL).*": 54.9,
                ".*SHOULDER.*": 61.0, ".*ELBOW.*": 61.0,
                ".*WAIST_YAW.*": 61.0, ".*HEAD_YAW.*": 52.0,
            },
            velocity_limit={
                ".*HIP_(PITCH|ROLL).*": 26.3, ".*KNEE_PITCH.*": 26.3,
                ".*HIP_YAW.*": 35.2, ".*ANKLE_(PITCH|ROLL).*": 35.2,
                ".*SHOULDER.*": 35.2, ".*ELBOW.*": 35.2,
                ".*WAIST_YAW.*": 35.2, ".*HEAD_YAW.*": 35.2,
            },
            velocity_limit_sim={
                ".*HIP_(PITCH|ROLL).*": 26.3, ".*KNEE_PITCH.*": 26.3,
                ".*HIP_YAW.*": 35.2, ".*ANKLE_(PITCH|ROLL).*": 35.2,
                ".*SHOULDER.*": 35.2, ".*ELBOW.*": 35.2,
                ".*WAIST_YAW.*": 35.2, ".*HEAD_YAW.*": 35.2,
            },
        )
    },
)

