"""The user's 10-DoF robot, loaded from the workspace model directory."""

import json
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets import ArticulationCfg


ROBOYI_MODEL_DIR = Path(__file__).resolve().parents[4] / "model"
with (ROBOYI_MODEL_DIR / "actuator_parameters.json").open() as stream:
    ROBOYI_MOTOR_PARAMETERS = json.load(stream)
with (ROBOYI_MODEL_DIR / "standing_pose.json").open() as stream:
    ROBOYI_STANDING_POSE = json.load(stream)
with (ROBOYI_MODEL_DIR / "mass_properties.json").open() as stream:
    ROBOYI_MASS_PROPERTIES = json.load(stream)

ROBOYI_JOINT_NAMES = list(ROBOYI_STANDING_POSE["new_zero_joint_positions_rad"])
ROBOYI_FOOT_HEIGHT = 0.025  # Foot link origin to the horizontal sole, in metres.
# Link masses and inertias are already scaled in the USD/MJCF/URDF assets.
# Use this factor only for training parameters expressed in mass or force units.
ROBOYI_MASS_SCALE = ROBOYI_MASS_PROPERTIES["mass_scale_from_original"]

ROBOYI_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(ROBOYI_MODEL_DIR / "isaacsim" / "roboyi.usda"),
        activate_contact_sensors=True,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_depenetration_velocity=1.0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, ROBOYI_STANDING_POSE["default_spawn_height_m"]),
        joint_pos=ROBOYI_STANDING_POSE["new_zero_joint_positions_rad"],
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=ROBOYI_JOINT_NAMES,
            effort_limit=ROBOYI_MOTOR_PARAMETERS["joint_effort_limit_Nm"],
            effort_limit_sim=ROBOYI_MOTOR_PARAMETERS["joint_effort_limit_Nm"],
            velocity_limit_sim=ROBOYI_MOTOR_PARAMETERS["joint_velocity_limit_rad_s"],
            stiffness=ROBOYI_MOTOR_PARAMETERS["kp_Nm_per_rad"],
            damping=ROBOYI_MOTOR_PARAMETERS["kv_Nm_s_per_rad"],
            armature=ROBOYI_MOTOR_PARAMETERS["joint_reflected_armature_kg_m2"],
            viscous_friction=ROBOYI_MOTOR_PARAMETERS["passive_damping_Nm_s_per_rad"],
            # Match the RPO command-delay randomization: 0/1/2 physics steps.
            # With the task's 5 ms physics step this is 0/5/10 ms, sampled per reset.
            min_delay=0,
            max_delay=2,
        ),
    },
)
