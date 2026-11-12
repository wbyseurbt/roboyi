"""Check knee direction, reward behavior and the actual roboyi environment."""

import argparse
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, default=Path("outputs/roboyi_gait_reward_verification/rewards.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import mujoco
import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robolab.tasks  # noqa: F401
from robolab.tasks.direct.base.mdp.rewards import (
    feet_contact_orientation_l2,
    feet_swing_clearance,
    feet_swing_knee_flexion,
    knee_hyperextension_l2,
)


def check_knee_direction():
    source = Path(__file__).resolve().parents[1] / "model/mujoco/roboyi.xml"
    model = mujoco.MjModel.from_xml_path(str(source))
    data = mujoco.MjData(model)
    result = []
    for side in ("Left", "Right"):
        for q in (-0.3, 0.3):
            mujoco.mj_resetData(model, data)
            data.qpos[model.jnt_qposadr[model.joint(side + "_knee").id]] = q
            mujoco.mj_forward(model, data)
            hip = data.xpos[model.body(side + "_thigh_link").id]
            knee = data.xpos[model.body(side + "_shank_link").id]
            ankle = data.xpos[model.body(side + "_foot_link").id]
            ratio = (knee[2] - hip[2]) / (ankle[2] - hip[2])
            forward = float(knee[0] - hip[0] - ratio * (ankle[0] - hip[0]))
            assert forward > 0 if q < 0 else forward < 0
            result.append({"joint": side + "_knee", "angle": q, "knee_forward_of_leg_chord_m": forward})
    return result


def check_reward_cases():
    count = 13
    forces = torch.zeros(count, 3, 2, 3)
    forces[..., 2] = 25.0
    forces[1:, :, 0, :] = 0.0
    forces[6] = 0.0  # Both feet airborne.
    forces[12, :, 0, 2] = 25.0
    forces[12, :, 1, :] = 0.0  # Right swing, left support.
    height = torch.zeros(count, 2)
    height[1:, 0] = torch.tensor([.01, .02, .03, .06, .03, .03, .03, .03, .03, .03, .03, 0.0])
    height[12, 1] = .03
    foot_velocity = torch.zeros(count, 2, 3)
    foot_velocity[1:, 0, 0] = .1
    foot_velocity[5] = 0.0  # A stationary raised foot must not earn reward.
    foot_velocity[12, 0] = 0.0
    foot_velocity[12, 1, 0] = .1
    command = torch.zeros(count, 3)
    command[:, 0] = .2
    command[7] = 0.0
    command[10, 0] = 0.0
    command[10, 2] = .3  # Turning also counts as movement.
    gravity = torch.tensor([[0.0, 0.0, -1.0]]).repeat(count, 1)
    gravity[9, 2] = 1.0
    rays = [torch.zeros(count, 4, 3), torch.zeros(count, 4, 3)]
    rays[0][8, :, 2] = float('nan')
    rays[0][11, :, 2] = float('inf')
    scanners = [SimpleNamespace(data=SimpleNamespace(pos_w=torch.zeros(count, 3), ray_hits_w=rays[i])) for i in range(2)]
    for i, scanner in enumerate(scanners):
        scanner.data.pos_w[:, 2] = .025 + height[:, i]
    robot = SimpleNamespace(data=SimpleNamespace(body_lin_vel_w=foot_velocity, projected_gravity_b=gravity))
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))
    env = SimpleNamespace(scene={"robot": robot, "contact": sensor, "left": scanners[0], "right": scanners[1]},
                          command_generator=SimpleNamespace(command=command))
    sensor_cfg = SceneEntityCfg("contact", body_ids=[0, 1])
    asset_cfg = SceneEntityCfg("robot", body_ids=[0, 1])
    clearance = feet_swing_clearance(env, sensor_cfg, asset_cfg, SceneEntityCfg("left"), SceneEntityCfg("right"))
    expected = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1], dtype=torch.float32)
    torch.testing.assert_close(clearance, expected)

    angle = np.pi / 18
    robot.data.joint_pos = torch.tensor([[0, 0], [-angle, -angle], [angle, 0], [2*angle, 0], [angle, angle], [-angle, angle]], dtype=torch.float32)
    knee_cost = knee_hyperextension_l2(env, SceneEntityCfg("robot", joint_ids=[0, 1]))
    torch.testing.assert_close(knee_cost, torch.tensor([0, 0, 1, 4, 2, 1], dtype=torch.float32))
    return {"clearance_cases": clearance.tolist(), "knee_cost_cases": knee_cost.tolist()}


def check_swing_knee_cases():
    labels = [
        "left_full", "half_knee", "quarter_knee", "straight_knee", "reverse_knee",
        "only_support_knee_bent", "double_support", "both_airborne_bent",
        "both_airborne_mixed_angles", "right_full", "above_target",
        "force_at_contact_threshold", "force_above_contact_threshold", "recent_contact",
    ]
    count = len(labels)
    target = np.pi / 9
    forces = torch.zeros(count, 3, 2, 3)
    forces[:, :, 1, 2] = 25.0
    forces[6, :, 0, 2] = 25.0
    forces[7:9] = 0.0
    forces[9] = forces[9].flip(1)
    forces[11, :, 0, 2] = 1.0
    forces[12, :, 0, 2] = 1.001
    forces[13, 0, 0, 2] = 25.0  # History still reports contact although the current force is zero.
    positions = torch.zeros(count, 2)
    positions[:, 0] = -target
    positions[1, 0] = -target / 2
    positions[2, 0] = -target / 4
    positions[3, 0] = 0.0
    positions[4, 0] = target
    positions[5] = torch.tensor([0.0, -target])
    positions[7] = -target
    positions[8] = torch.tensor([-target / 2, -target])
    positions[9] = positions[9].flip(0)
    positions[10, 0] = -2 * target
    # Deliberately omit scanners, foot velocities, commands and body posture:
    # the simplified reward must depend only on contacts and knee angles.
    robot = SimpleNamespace(data=SimpleNamespace(joint_pos=positions))
    env = SimpleNamespace(scene={
        "robot": robot, "contact": SimpleNamespace(data=SimpleNamespace(net_forces_w_history=forces))})
    def calculate(order):
        return feet_swing_knee_flexion(
            env, SceneEntityCfg("contact", body_ids=order),
            SceneEntityCfg("robot", joint_ids=order), target_flexion=target)
    expected = torch.tensor([1, .5, .25, 0, 0, 0, 0, 2, 1.5, 1, 1, 1, 0, 0])
    actual = calculate([0, 1])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(calculate([1, 0]), expected)
    return dict(zip(labels, actual.tolist()))


def check_environment():
    cfg = load_cfg_from_registry("Roboyi-Flat", "env_cfg_entry_point")
    cfg.scene.num_envs = 16
    cfg.seed = 42
    if args.device is not None:
        cfg.sim.device = args.device
    env = gym.make("Roboyi-Flat", cfg=cfg)
    try:
        base = env.unwrapped
        env.reset()
        term = base.reward_manager.get_term_cfg("knee_hyperextension")
        assert term.weight == -5.0
        assert "knee_hyperextension" in base.reward_manager.active_terms
        assert base.reward_manager.get_term_cfg("feet_height").weight == 0.3
        assert base.reward_manager.get_term_cfg("feet_air_time").weight == 0.25
        assert base.reward_manager.get_term_cfg("feet_air_time").params["threshold"] == 0.6
        assert base.reward_manager.get_term_cfg("track_lin_vel_xy_exp").params["std"] == 0.5
        assert base.reward_manager.get_term_cfg("feet_contact_orientation_l2").weight == -0.2
        swing_term = base.reward_manager.get_term_cfg("swing_knee_flexion")
        assert swing_term.weight == 0.1
        assert swing_term.params["target_flexion"] == np.deg2rad(20.0)
        assert set(swing_term.params) == {"sensor_cfg", "knee_cfg", "target_flexion", "flexion_sign", "straight_angle"}
        assert "knee_distance" not in base.reward_manager.active_terms
        feet_term = base.reward_manager.get_term_cfg("feet_distance")
        assert (feet_term.params["min"], feet_term.params["max"]) == (0.27, 0.33)
        assert [base.robot.joint_names[i] for i in swing_term.params["knee_cfg"].joint_ids] == ["Left_knee", "Right_knee"]
        assert [base.contact_sensor.body_names[i] for i in swing_term.params["sensor_cfg"].body_ids] == ["Left_foot_link", "Right_foot_link"]
        knee_ids = term.params["asset_cfg"].joint_ids
        assert [base.robot.joint_names[i] for i in knee_ids] == ["Left_knee", "Right_knee"]
        q = base.robot.data.joint_pos.clone()
        q[:, knee_ids] = -np.pi / 18
        q[1, knee_ids] = torch.tensor([np.pi / 18, 0.0], device=base.device)
        q[2, knee_ids] = np.pi / 18
        base.robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        costs = term.func(base, **term.params)
        torch.testing.assert_close(costs[:3], torch.tensor([0., 1., 2.], device=base.device))
        env.reset()
        with torch.inference_mode():
            for _ in range(64):
                actions = .5 * torch.randn(16, 10, device=base.device)
                obs, reward, _, _, _ = env.step(actions)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                swing_reward = swing_term.func(base, **swing_term.params)
                assert torch.isfinite(swing_reward).all()
                assert ((swing_reward >= 0) & (swing_reward <= 2)).all()
        return {"device": str(base.device), "steps": 64, "num_envs": 16,
                "resolved_knee_indices": knee_ids, "actual_joint_costs": costs[:3].cpu().tolist(),
                "active_rewards": base.reward_manager.active_terms, "finite_rewards_observations": True}
    finally:
        env.close()


def check_contact_orientation():
    # Flat contact, tilted contact, identical tilt in the air, yaw-only contact,
    # upside-down contact, then the right foot contacting instead of the left.
    quats = torch.zeros(6, 2, 4)
    quats[..., 0] = 1.0
    angle = np.pi / 6
    for i in (1, 2):
        quats[i, 0] = torch.tensor([np.cos(angle/2), np.sin(angle/2), 0.0, 0.0])
    quats[3, 0] = torch.tensor([np.cos(angle/2), 0.0, 0.0, np.sin(angle/2)])
    quats[4, 0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    quats[5, 1] = quats[1, 0]
    forces = torch.zeros(6, 2, 3)
    forces[:, 0, 2] = 25.0
    forces[2] = 0.0
    forces[5, 0] = 0.0
    forces[5, 1, 2] = 25.0
    data = SimpleNamespace(body_quat_w=quats, GRAVITY_VEC_W=torch.tensor([[0., 0., -1.]]).repeat(6, 1))
    env = SimpleNamespace(scene={"robot": SimpleNamespace(data=data), "contact": SimpleNamespace(data=SimpleNamespace(net_forces_w=forces))})
    costs = feet_contact_orientation_l2(env, SceneEntityCfg("contact", body_ids=[0, 1]), SceneEntityCfg("robot", body_ids=[0, 1]))
    tilt_cost = 2.0 * (1.0 - np.cos(angle))
    torch.testing.assert_close(costs, torch.tensor([0., tilt_cost, 0., 0., 4., tilt_cost], dtype=torch.float32))
    return costs.tolist()


if __name__ == "__main__":
    try:
        result = {"knee_direction": check_knee_direction(), "reward_cases": check_reward_cases(),
                  "swing_knee_cases": check_swing_knee_cases(), "contact_orientation_cases": check_contact_orientation(),
                  "runtime": check_environment(), "passed": True}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print("GAIT_REWARDS_VERIFIED", json.dumps(result), flush=True)
    except Exception:
        traceback.print_exc()
        raise
    finally:
        app.close(wait_for_replicator=False)
