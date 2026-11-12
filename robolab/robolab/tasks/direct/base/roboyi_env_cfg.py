"""Flat-ground PPO walking task for the 10-DoF roboyi robot."""

import math

import isaaclab.sim as sim_utils
from isaaclab.managers import RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass

from robolab.assets.robots.roboyi import ROBOYI_CFG, ROBOYI_FOOT_HEIGHT, ROBOYI_JOINT_NAMES, ROBOYI_MASS_SCALE
from . import mdp
from .base_config import BaseEnvCfg, RewardCfg
from .scene_cfg import SceneCfg


FEET = ["Left_foot_link", "Right_foot_link"]
KNEES = ["Left_shank_link", "Right_shank_link"]
KNEE_JOINTS = ["Left_knee", "Right_knee"]


@configclass
class RoboyiRewardCfg(RewardCfg):
    track_lin_vel_xy_exp = RewTerm(func=mdp.track_lin_vel_xy_yaw_frame_exp, weight=1.0, params={"std": 0.5})
    track_ang_vel_z_exp = RewTerm(func=mdp.track_ang_vel_z_world_exp, weight=1.0, params={"std": 0.5})
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.2)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.1)
    energy = RewTerm(func=mdp.energy, weight=-1e-4)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-2e-4)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-2e-2)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=-2e-2)
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names="(?!.*_foot_link$).*")},
    )
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped, weight=0.25,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET, preserve_order=True),
                "threshold": 0.5},
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide, weight=-0.3,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True)},
    )
    feet_force = RewTerm(
        func=mdp.body_force, weight=-3e-3 / ROBOYI_MASS_SCALE,
        # Preserve the force threshold and penalty relative to body weight:
        # 5.051 -> 7.072 kg; threshold/cap 100 -> 140 N at mass scale 1.4.
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET),
                "threshold": 100.0 * ROBOYI_MASS_SCALE, "max_reward": 100.0 * ROBOYI_MASS_SCALE},
    )
    feet_distance = RewTerm(
        func=mdp.body_distance_y, weight=0.4,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET), "min": 0.26, "max": 0.33},
    )
    # knee_distance = RewTerm(
    #     func=mdp.body_distance_y, weight=0.1,
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=KNEES), "min": 0.12, "max": 0.28},
    # )
    feet_stumble = RewTerm(
        func=mdp.feet_stumble, weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET)},
    )
    # Foot orientation is still meaningful without ankle roll: hip roll and body
    # motion also change it. It does not require a separate ankle-roll joint.
    feet_orientation_l2 = RewTerm(
        func=mdp.body_orientation_l2, weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=FEET)},
    )
    feet_contact_orientation_l2 = RewTerm(
        func=mdp.feet_contact_orientation_l2, weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True)},
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
    # Positive knee angles are reverse bending in this robot (verified with FK).
    # At +10 deg a single knee costs 5 reward units/s; at +20 deg it costs 20.
    # The two speed-tracking terms together can earn at most 2 units/s.
    knee_hyperextension = RewTerm(
        func=mdp.knee_hyperextension_l2, weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINTS, preserve_order=True),
                "hyperextension_sign": 1.0, "straight_angle": 0.0, "angle_scale": math.radians(10.0)},
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(Left|Right)_yaw", "(Left|Right)_roll"])},
    )
    # roboyi has no torso, shoulder or elbow joints. The corresponding RPO rewards
    # are intentionally commented out instead of selecting nonexistent joints:
    # joint_deviation_torso = RewTerm(func=mdp.joint_deviation_l1, weight=-1.0, ...)
    # joint_deviation_arms = RewTerm(func=mdp.joint_deviation_l1, weight=-0.06, ...)
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["(Left|Right)_pitch", "(Left|Right)_knee", "(Left|Right)_ankle"])},
        # No ".*_ankle_roll.*" term: each foot only has ankle pitch.
    )
    feet_contact_without_cmd = RewTerm(
        func=mdp.feet_contact_without_cmd, weight=0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET)},
    )
    upward = RewTerm(func=mdp.upward, weight=0.4)
    stand_still = RewTerm(
        func=mdp.stand_still, weight=-0.2,
        params={"pos_cfg": SceneEntityCfg("robot", joint_names=ROBOYI_JOINT_NAMES),
                "vel_cfg": SceneEntityCfg("robot", joint_names=ROBOYI_JOINT_NAMES),
                "pos_weight": 1.0, "vel_weight": 0.04},
    )
    feet_height = RewTerm(
        func=mdp.feet_swing_clearance, weight=0.3,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET, preserve_order=True),
                "asset_cfg": SceneEntityCfg("robot", body_names=FEET, preserve_order=True),
                "sensor_cfg1": SceneEntityCfg("left_feet_scanner"),
                "sensor_cfg2": SceneEntityCfg("right_feet_scanner"),
                "ankle_height": ROBOYI_FOOT_HEIGHT, "threshold": 0.02, "swing_speed": 0.1},
    )
    swing_knee_flexion = RewTerm(
        func=mdp.feet_swing_knee_flexion, weight=0.5,
        params={"sensor_cfg": SceneEntityCfg("contact_sensor", body_names=FEET, preserve_order=True),
                "knee_cfg": SceneEntityCfg("robot", joint_names=KNEE_JOINTS, preserve_order=True),
                "target_flexion": math.radians(25.0),
                "flexion_sign": -1.0, "straight_angle": 0.0},
    )


@configclass
class RoboyiFlatEnvCfg(BaseEnvCfg):
    reward = RoboyiRewardCfg()

    def __post_init__(self):
        super().__post_init__()
        self.action_space = len(ROBOYI_JOINT_NAMES)
        single_actor_obs = 9 + 3 * self.action_space
        single_critic_obs = single_actor_obs + 15 + 2 * self.action_space
        self.observation_space = single_actor_obs * self.robot.actor_obs_history_length
        self.state_space = single_critic_obs * self.robot.critic_obs_history_length
        self.scene_context.num_envs = 1024
        self.scene_context.robot = ROBOYI_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene_context.robot_body_path = "Geometry"
        self.scene_context.left_foot_body_name = FEET[0]
        self.scene_context.right_foot_body_name = FEET[1]
        self.scene_context.height_scanner.prim_body_name = "base_link"
        self.scene_context.terrain_type = "plane"
        self.scene_context.terrain_generator = None
        self.scene_context.height_scanner.enable_height_scan = False
        self.scene = SceneCfg(self.scene_context, self.sim.dt, self.decimation * self.sim.dt)
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.4, 0.4))
        self.scene.sky_light.spawn.texture_file = None
        self.scene.left_feet_scanner.debug_vis = False
        self.scene.right_feet_scanner.debug_vis = False
        self.commands.debug_vis = True
        # Initial task: forward/backward walking and turning, no lateral command.
        self.commands.ranges.lin_vel_x = (-0.3, 0.6)
        self.commands.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.ranges.ang_vel_z = (-0.5, 0.5)
        self.robot.feet_body_names = FEET
        self.robot.feet_height_offset = ROBOYI_FOOT_HEIGHT
        self.robot.action_scale = 0.15
        self.robot.clip_joint_targets = True
        self.robot.terminate_contacts_body_names = ["base_link", "(Left|Right)_hip_pitch_link", "(Left|Right)_hip_roll_link", "(Left|Right)_thigh_link"]
        self.robot.terminate_base_height = 0.20
        self.robot.terminate_base_orientation = math.pi / 3
        self.events.add_base_mass.params["asset_cfg"].body_names = ["base_link"]
        self.events.add_base_mass.params["mass_distribution_params"] = (
            -0.2 * ROBOYI_MASS_SCALE, 0.2 * ROBOYI_MASS_SCALE,
        )
        self.events.randomize_rigid_body_com.params["asset_cfg"].body_names = ["base_link"]
        self.events.randomize_rigid_body_com.params["com_range"] = {
            "x": (-0.005, 0.005), "y": (-0.005, 0.005), "z": (-0.01, 0.01),
        }
        self.events.scale_link_mass.params["asset_cfg"].body_names = ["(Left|Right)_.*_link"]
        self.events.scale_actuator_gains.params["asset_cfg"].joint_names = ROBOYI_JOINT_NAMES
        self.events.scale_joint_parameters.params["asset_cfg"].joint_names = ROBOYI_JOINT_NAMES
        self.events.reset_robot_joints.params["position_range"] = (-0.05, 0.05)
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.1, 0.1), "y": (-0.1, 0.1), "z": (0.0, 0.0),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.1, 0.1),
        }
        self.events.push_robot.params["velocity_range"] = {
            "x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (0.0, 0.0),
            "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2),
        }
        self.noise.noise_scales.joint_vel = 1.0
        self.noise.noise_scales.joint_pos = 0.01
