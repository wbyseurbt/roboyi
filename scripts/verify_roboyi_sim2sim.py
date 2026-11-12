"""Check Roboyi sim2sim inference, coordinates, history and actuator wiring.

./scripts/python.sh scripts/verify_roboyi_sim2sim.py --checkpoint /path/to/model_N.pt
"""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("sim2sim_roboyi", ROOT / "robolab/scripts/mujoco/sim2sim_roboyi.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--viewer", action="store_true", help="Also send real keyboard events (requires X11/xdotool)")
    args = parser.parse_args()
    torch.set_num_threads(1)
    env, agent = module.load_config(args.checkpoint.parent / "params")
    with tempfile.TemporaryDirectory() as directory:
        exported = Path(directory) / "policy.pt"
        options = SimpleNamespace(checkpoint=args.checkpoint, load_model=None, export_policy=exported)
        policy, iteration = module.load_policy(options, env, agent)
        jit = torch.jit.load(str(exported)).eval()
        from rsl_rl.modules import ActorCritic
        from tensordict import TensorDict

        torch.manual_seed(123)
        batch = TensorDict({"policy": torch.randn(8, env["observation_space"]),
                            "critic": torch.randn(8, env["state_space"])}, batch_size=[8])
        cfg = dict(agent["policy"])
        cfg.pop("class_name")
        canonical = ActorCritic(batch, agent["obs_groups"], 10, **cfg).eval()
        canonical.load_state_dict(torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model_state_dict"])
        with torch.inference_mode():
            reference = canonical.act_inference(batch)
            torch.testing.assert_close(policy(batch["policy"]), reference, rtol=0, atol=0)
            torch.testing.assert_close(jit(batch["policy"]), reference, rtol=0, atol=0)

        sim = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", env, agent, policy)
        assert sim.qadr.tolist() == [7, 12, 8, 13, 9, 14, 10, 15, 11, 16]
        assert sim.aids.tolist() == [0, 5, 1, 6, 2, 7, 3, 8, 4, 9]
        np.testing.assert_allclose(sim.state()[1], [0, 0, -1], atol=1e-12)
        obs, _ = sim.infer(np.zeros(3))
        frames = obs.reshape(-1, 39)
        np.testing.assert_array_equal(frames, np.tile(frames[0], (len(frames), 1)))
        old = frames.copy()
        previous = sim.previous_action.copy()
        obs, _ = sim.infer(np.array([.1, 0, 0]))
        np.testing.assert_array_equal(obs.reshape(-1, 39)[:-1], old[1:])
        np.testing.assert_array_equal(obs[0, -10:], previous)
        np.testing.assert_allclose(obs.reshape(-1, 39)[-1, 6:9], [.1, 0, 0])

        # Yaw rotates world +X into body -Y. Angular velocity must stay in body axes.
        sim.reset()
        sim.data.qpos[3:7] = [np.sqrt(.5), 0, 0, np.sqrt(.5)]
        sim.data.qvel[3:6] = [.2, -.3, .4]
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose(sim.state()[0][:3], [.2, -.3, .4], atol=1e-12)
        np.testing.assert_allclose(sim.state()[1], [0, 0, -1], atol=1e-12)
        # Roll 90 degrees: body-frame gravity is -Y (also checks wxyz convention).
        sim.data.qpos[3:7] = [np.sqrt(.5), np.sqrt(.5), 0, 0]
        mujoco.mj_forward(sim.model, sim.data)
        np.testing.assert_allclose(sim.state()[1], [0, -1, 0], atol=1e-12)
        assert sim.failure() == "base_orientation"

        # Exercise clipping with actions beyond both the wrapper and roll limits.
        class SaturatedPolicy(torch.nn.Module):
            def forward(self, obs):
                return torch.tensor([[100., -100.] * 5])

        saturated = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml",
                                      env, agent, SaturatedPolicy())
        saturated.infer(np.zeros(3))
        clipped = np.array([3., -3.] * 5)
        np.testing.assert_array_equal(saturated.previous_action, clipped)
        np.testing.assert_allclose(saturated.target, np.clip(saturated.q0 + .15 * clipped,
                                                            saturated.lower, saturated.upper))
        assert saturated.target[2] == saturated.upper[2]
        assert saturated.target[3] == saturated.lower[3]

        keys = module.Keyboard([0, 0, 0], env["commands"]["ranges"])
        keys.callback(ord("A"))
        assert keys.read()[0][1] == 0  # Lateral commands were excluded in training.
        keys.callback(ord("W"))
        assert keys.read()[0][0] == .05
        keys.callback(ord("R"))
        assert keys.read()[1] is True and keys.read()[1] is False
        keys.callback(32)
        np.testing.assert_array_equal(keys.read()[0], np.zeros(3))

        ranges = module.command_ranges(env, 1.0)
        assert ranges["lin_vel_x"] == (env["commands"]["ranges"]["lin_vel_x"][0], 1.0)
        faster_keys = module.Keyboard([.4, 0, 0], ranges)
        faster_keys.callback(ord("W"))
        np.testing.assert_allclose(faster_keys.read()[0], [.45, 0, 0])
        limited_keys = module.Keyboard([1.45, 0, 0], module.command_ranges(env, 1.5))
        limited_keys.callback(ord("W"))
        limited_keys.callback(ord("W"))
        assert limited_keys.read()[0][0] == 1.5
        for invalid in (0, -1, float("inf"), float("nan")):
            try:
                module.command_ranges(env, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Invalid command limit accepted")

        if args.viewer:
            with module.RobotViewer(sim.model, sim.data, faster_keys) as window:
                window.sync()
                raw = window.viewer
                before = (raw.vopt.flags.copy(), raw.scn.flags.copy(), raw.model.geom_rgba.copy(),
                          raw.vopt.frame, raw._paused, raw._render_every_frame)
                xid = window.glfw.get_x11_window(raw.window)

                def send_key(key):
                    subprocess.run(["xdotool", "key", "--window", str(xid), key], check=True)
                    window.sync()

                send_key("w")
                np.testing.assert_allclose(faster_keys.read()[0], [.5, 0, 0])
                for key in ("s", "a", "d", "q", "e", "r", "space"):
                    send_key(key)
                np.testing.assert_array_equal(faster_keys.read()[0], np.zeros(3))
                after = (raw.vopt.flags, raw.scn.flags, raw.model.geom_rgba,
                         raw.vopt.frame, raw._paused, raw._render_every_frame)
                for old, new in zip(before, after):
                    np.testing.assert_array_equal(old, new)
                send_key("Escape")
                assert not window.is_running()

        # Same observation must yield the same targets through either load path.
        sim.reset()
        other = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", env, agent, jit)
        # Compare load paths under the same sampled per-episode delay.
        other.command_delay_steps = sim.command_delay_steps
        other.command_delay_seconds = sim.command_delay_seconds
        for _ in range(10):
            sim.infer(np.array([.1, 0, 0]))
            other.infer(np.array([.1, 0, 0]))
            np.testing.assert_array_equal(sim.target, other.target)
            for _ in range(sim.substeps):
                q_before = sim.data.qpos[sim.qadr].copy()
                dq_before = sim.data.qvel[sim.vadr].copy()
                sim.physics_step()
                other.physics_step()
                expected = np.clip(sim.kp * (sim.applied_target - q_before)
                                   + sim.kd * (sim.dq0 - dq_before), -sim.effort, sim.effort)
                np.testing.assert_allclose(sim.data.actuator_force[sim.aids], expected, rtol=0, atol=1e-12)
                np.testing.assert_array_equal(sim.data.qpos, other.data.qpos)
                assert (sim.target >= sim.lower).all() and (sim.target <= sim.upper).all()
        sim.reset()
        reset_obs, _ = sim.infer(np.zeros(3))
        assert sim.data.time == 0
        np.testing.assert_array_equal(reset_obs[0, -10:], np.zeros(10))
        np.testing.assert_array_equal(sim.history, np.tile(sim.history[0], (len(sim.history), 1)))

        # Isolated impulse: include floating-base coupling, preserve unactuated
        # generalized momentum, dissipate energy, and integrate a consistent pose.
        limited = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", env, agent, policy)
        limited.model.opt.gravity[:] = 0
        limited.model.geom_contype[:] = 0
        limited.model.geom_conaffinity[:] = 0
        limited.data.qpos[2] = 2
        limited.data.qvel[:] = np.linspace(-.5, .5, limited.model.nv)
        limited.data.qvel[limited.vadr] = [7., -8.] * 5
        mujoco.mj_forward(limited.model, limited.data)
        inertia = np.empty((limited.model.nv, limited.model.nv))
        mujoco.mj_fullM(limited.model, inertia, limited.data.qM)
        free = limited.data.qvel.copy()
        position = limited.data.qpos.copy()
        limited.apply_velocity_limit(position)
        constrained = limited.data.qvel.copy()
        assert np.max(np.abs(constrained[limited.vadr])) <= 3 + 1e-8
        np.testing.assert_allclose(inertia[:6] @ (constrained - free), 0, atol=1e-12)
        assert np.linalg.norm(constrained[:6] - free[:6]) > 1e-3
        assert constrained @ inertia @ constrained < free @ inertia @ free
        generalized_impulse = np.zeros(limited.model.nv)
        generalized_impulse[limited.vadr] = limited.last_limit_impulse
        np.testing.assert_allclose(inertia @ (constrained - free), generalized_impulse, atol=1e-12)
        expected_position = position.copy()
        mujoco.mj_integratePos(limited.model, expected_position, constrained, limited.dt)
        np.testing.assert_array_equal(limited.data.qpos, expected_position)

        # Below the bound, the limiter must not change the state at all.
        limited.data.qvel[:] = 0
        limited.data.qvel[limited.vadr] = np.linspace(-2.9, 2.9, 10)
        free = limited.data.qvel.copy()
        position = limited.data.qpos.copy()
        limited.apply_velocity_limit(position)
        np.testing.assert_array_equal(limited.data.qvel, free)
        np.testing.assert_array_equal(limited.data.qpos, position)

        # Full physics steps: strong commands in both directions, including
        # recovery from an already excessive initial joint velocity.
        peak = 0.
        active_steps = 0
        finer_dt = .001 if env["scene_context"]["robot"]["actuators"]["legs"].get("max_delay", 0) else .002
        for physics_dt in (.005, finer_dt):
            stress = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", env, agent, policy,
                                      physics_dt=physics_dt)
            stress.model.opt.gravity[:] = 0
            stress.model.geom_contype[:] = 0
            stress.model.geom_conaffinity[:] = 0
            stress.data.qpos[2] = 2
            stress.data.qvel[stress.vadr] = [8., -8.] * 5
            mujoco.mj_forward(stress.model, stress.data)
            for step in range(500):
                stress.target = stress.upper.copy() if step < 250 else stress.lower.copy()
                stress.physics_step()
                peak = max(peak, float(np.max(np.abs(stress.data.qvel[stress.vadr]))))
                active_steps += int(np.any(stress.last_limit_impulse))
                assert np.all(np.abs(stress.data.qvel[stress.vadr]) <= stress.velocity_limit + 1e-8)
                assert np.all(np.abs(stress.data.actuator_force[stress.aids]) <= stress.effort + 1e-10)
                assert np.isfinite(stress.data.qpos).all()
                assert not np.any(stress.data.warning.number)
        assert active_steps > 0

        unbounded = module.RoboyiSim(mujoco, ROOT / "model/mujoco/standing_scene.xml", env, agent, policy,
                                     enforce_velocity_limit=False)
        unbounded.model.opt.gravity[:] = 0
        unbounded.data.qpos[2] = 2
        unbounded.data.qvel[unbounded.vadr] = 8
        mujoco.mj_forward(unbounded.model, unbounded.data)
        unbounded.physics_step()
        assert np.max(np.abs(unbounded.data.qvel[unbounded.vadr])) > 3
        assert not np.any(unbounded.last_limit_impulse)
        print(json.dumps({"passed": True, "checkpoint_iteration": iteration,
                          "velocity_limit_stress_peak_rad_s": peak,
                          "velocity_limit_stress_active_steps": active_steps,
                          "checks": ["canonical_actor_and_normalizer_parity", "torchscript_parity",
                                     "joint_mapping", "history_initialization_and_order", "previous_action",
                                     "body_frame_angular_velocity", "projected_gravity_wxyz", "fall_detection",
                                     "explicit_pd_no_double_control", "action_clipping_and_joint_target_limits",
                                     "keyboard_commands_and_reset", "increased_and_configurable_vx_limit",
                                     *(["real_glfw_keys_preserve_rendering_and_colors"] if args.viewer else []), "reset",
                                     "mass_weighted_velocity_limit_momentum_and_energy", "velocity_limit_pose_consistency",
                                     "velocity_limit_inactive_state_identity", "velocity_limit_stress_both_directions",
                                     "velocity_limit_two_physics_steps", "actuator_effort_limit_preserved",
                                     "velocity_limit_disable_for_comparison",
                                     "identical_checkpoint_and_jit_trajectories"]}, indent=2))


if __name__ == "__main__":
    main()
