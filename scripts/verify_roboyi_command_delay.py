"""Compare native Isaac Lab command delay against the MuJoCo implementation."""

import argparse
import copy
import importlib.util
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--config-dir", type=Path, required=True)
parser.add_argument("--output", type=Path, default=Path("outputs/roboyi_delay_verification/delay_semantics.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import mujoco
import numpy as np
import torch
from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils.types import ArticulationActions

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("roboyi_sim2sim_delay", ROOT / "robolab/scripts/mujoco/sim2sim_roboyi.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ZeroPolicy(torch.nn.Module):
    def forward(self, obs):
        return torch.zeros((obs.shape[0], 10))


def main():
    env, agent = module.load_config(args.config_dir)
    assert module.actuator_delay_range(env["scene_context"]["robot"]["actuators"]["legs"]) == (0, 2)
    cfg = DelayedPDActuatorCfg(joint_names_expr=["test_joint"], min_delay=0, max_delay=2,
                              stiffness=100.0, damping=2.0, effort_limit=20.0)
    native = DelayedPDActuator(cfg, joint_names=["test_joint"], joint_ids=slice(None), num_envs=3, device="cpu")
    native.reset(None)
    for buffer in (native.positions_delay_buffer, native.velocities_delay_buffer, native.efforts_delay_buffer):
        buffer.set_time_lag(torch.tensor([0, 1, 2], dtype=torch.int))
    simulators = []
    for lag in (0, 1, 2):
        settings = copy.deepcopy(env)
        settings["scene_context"]["robot"]["actuators"]["legs"].update(min_delay=lag, max_delay=lag)
        sim = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", settings, agent, ZeroPolicy())
        simulators.append(sim)
    sequences = []
    for step in range(7):
        target = torch.full((3, 1), 0.01 * step)
        # Current feedback changes every physics step and must NOT be delayed.
        q = torch.full((3, 1), 0.001 * step)
        dq = torch.full((3, 1), 0.02 * step)
        output = native.compute(ArticulationActions(joint_positions=target, joint_velocities=torch.zeros_like(target),
                                                   joint_efforts=torch.zeros_like(target)), q, dq)
        applied_targets = (output.joint_efforts + 2.0 * dq) / 100.0 + q
        expected = torch.tensor([[0.01 * max(0, step - lag)] for lag in (0, 1, 2)])
        torch.testing.assert_close(applied_targets, expected)
        for lag, sim in enumerate(simulators):
            sim.target[:] = 0.01 * step
            expected_torque = np.clip(sim.kp * (float(expected[lag, 0]) - sim.data.qpos[sim.qadr])
                                      + sim.kd * (sim.dq0 - sim.data.qvel[sim.vadr]), -sim.effort, sim.effort)
            sim.physics_step()
            np.testing.assert_allclose(sim.applied_target, expected[lag, 0].item(), atol=1e-8)
            np.testing.assert_allclose(sim.data.actuator_force[sim.aids], expected_torque, atol=1e-6)
        sequences.append(applied_targets.flatten().tolist())
    for sim in simulators:
        sim.reset()
        sim.target[:] = 0.1
        sim.physics_step()
        np.testing.assert_allclose(sim.applied_target, 0.1)  # Oldest available sample during warmup.
    settings = copy.deepcopy(env)
    settings["scene_context"]["robot"]["actuators"]["legs"].update(min_delay=1, max_delay=1)
    fine = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", settings, agent, ZeroPolicy(), physics_dt=.001)
    assert fine.command_delay_seconds == .005
    for step in range(10):
        fine.target[:] = 0.0 if step < 3 else 0.05
        fine.physics_step()
        np.testing.assert_allclose(fine.applied_target, 0.0 if step < 8 else 0.05)
    try:
        module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", settings, agent, ZeroPolicy(), physics_dt=.002)
    except ValueError as exc:
        assert "training physics step" in str(exc)
    else:
        raise AssertionError("An unrepresentable 5 ms delay was accepted on a 2 ms physics grid")
    settings["scene_context"]["robot"]["actuators"]["legs"].update(min_delay=0, max_delay=0)
    legacy = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", settings, agent, ZeroPolicy(), physics_dt=.002)
    legacy.target[:] = .05
    legacy.physics_step()
    np.testing.assert_allclose(legacy.applied_target, .05)
    result = {"passed": True, "native_isaac_vs_mujoco_applied_targets": sequences,
              "tested_delay_ms": [0, 5, 10], "checks": ["latest_feedback_not_delayed", "same_delayed_position_targets",
              "mujoco_pd_effort_uses_delayed_target", "reset_buffer_warmup", "5ms_delay_preserved_at_1ms_dt",
              "inexact_delay_grid_rejected", "legacy_zero_delay_at_2ms_dt"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print("COMMAND_DELAY_VERIFIED", json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close(wait_for_replicator=False)
