"""Run a trained Roboyi-Flat policy in MuJoCo, without starting Isaac Sim.

See SIM2SIM_CN.md at the workspace root for commands and known physics gaps.
"""

from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[3]
# Runtime articulation order verified in outputs/roboyi_ppo_verification/runtime.json.
# This is the policy's order, not the order of init_state.joint_pos in YAML.
POLICY_JOINTS = [f"{side}_{joint}" for joint in ("pitch", "roll", "yaw", "knee", "ankle")
                 for side in ("Left", "Right")]


class ConfigLoader(yaml.SafeLoader):
    """Read Isaac Lab's YAML tags as plain data; never import/execute tags."""


def plain_tag(loader, _tag, node):
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node)
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node)
    return loader.construct_scalar(node)


ConfigLoader.add_multi_constructor("tag:yaml.org,2002:python/", plain_tag)


def load_config(directory):
    configs = []
    for name in ("env.yaml", "agent.yaml"):
        path = directory / name
        if not path.is_file():
            raise ValueError(f"Missing {path}; pass --config-dir pointing to the run's params directory")
        configs.append(yaml.load(path.read_text(), Loader=ConfigLoader))
    env, agent = configs
    asset = env["scene_context"]["robot"]
    if set(asset["init_state"]["joint_pos"]) != set(POLICY_JOINTS):
        raise ValueError("This script requires the 10-joint Roboyi-Flat configuration")
    if env["observation_space"] != 39 * env["robot"]["actor_obs_history_length"]:
        raise ValueError("Expected 39 observations per frame; height scans/other tasks are unsupported")
    if env["scene_context"]["height_scanner"]["enable_height_scan"]:
        raise ValueError("Height-scan policies are unsupported")
    if agent["policy"]["class_name"] != "ActorCritic":
        raise ValueError("Only the feed-forward Roboyi ActorCritic policy is supported")
    actuator_delay_range(asset["actuators"]["legs"])
    return env, agent


def actuator_delay_range(motor):
    """Resolve delay in saved training physics steps (zero for legacy runs)."""
    low, high = motor.get("min_delay", 0), motor.get("max_delay", 0)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in (low, high)):
        raise ValueError("Actuator delay bounds must be integer physics steps")
    if not 0 <= low <= high:
        raise ValueError("Expected 0 <= min_delay <= max_delay")
    return low, high


def joint_values(value):
    """Resolve scalar or regex-keyed Isaac Lab parameters in policy order."""
    if isinstance(value, dict):
        result = []
        for name in POLICY_JOINTS:
            matches = [v for pattern, v in value.items() if re.fullmatch(pattern, name)]
            if len(matches) != 1:
                raise ValueError(f"Expected one parameter match for {name}, got {matches}")
            result.append(matches[0])
        return np.asarray(result, dtype=np.float64)
    return np.full(10, value, dtype=np.float64)


class InferencePolicy(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.actor = policy.actor
        self.normalizer = policy.actor_obs_normalizer

    def forward(self, obs):
        return self.actor(self.normalizer(obs))


def load_policy(args, env, agent):
    if args.checkpoint:
        # Use this workspace's RSL-RL, without importing robolab/Isaac Lab.
        sys.path.insert(0, str(ROOT / "rsl_rl"))
        from rsl_rl.modules import ActorCritic
        from tensordict import TensorDict

        cfg = dict(agent["policy"])
        cfg.pop("class_name")
        if cfg.get("state_dependent_std", False):
            raise ValueError("State-dependent action standard deviation is unsupported")
        obs = TensorDict({"policy": torch.zeros(1, env["observation_space"]),
                          "critic": torch.zeros(1, env["state_space"])}, batch_size=[1])
        network = ActorCritic(obs, agent["obs_groups"], 10, **cfg)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        network.load_state_dict(checkpoint["model_state_dict"], strict=True)
        network.eval()
        policy = InferencePolicy(network).eval()
        iteration = int(checkpoint["iter"])
        if args.export_policy:
            args.export_policy.parent.mkdir(parents=True, exist_ok=True)
            torch.jit.script(policy).save(str(args.export_policy))
            print(f"Exported actor + normalizer: {args.export_policy}")
    else:
        if args.export_policy:
            raise ValueError("--export-policy requires --checkpoint")
        policy = torch.jit.load(str(args.load_model), map_location="cpu").eval()
        iteration = None
    with torch.inference_mode():
        output = policy(torch.zeros(1, env["observation_space"]))
    if output.shape != (1, 10) or not torch.isfinite(output).all():
        raise ValueError("Policy must produce 10 finite actions for the configured observation size")
    if hasattr(policy, "reset"):
        policy.reset()
    return policy, iteration


class RoboyiSim:
    def __init__(self, mj, model_path, env, agent, policy, physics_dt=None, enforce_velocity_limit=True):
        self.mj, self.env, self.policy = mj, env, policy
        self.model = mj.MjModel.from_xml_path(str(model_path))
        model = self.model
        self.data = mj.MjData(model)
        self.dt = physics_dt or env["sim"]["dt"]
        self.control_dt = env["sim"]["dt"] * env["decimation"]
        self.substeps = round(self.control_dt / self.dt)
        if self.substeps < 1 or not math.isclose(self.substeps * self.dt, self.control_dt, abs_tol=1e-10):
            raise ValueError("--physics-dt must divide the policy period exactly (normally 0.02 s)")
        model.opt.timestep = self.dt
        model.opt.gravity[:] = env["sim"]["gravity"]
        self.body = model.body("base_link").id
        self.root_joint = model.joint("root").id
        if model.jnt_type[self.root_joint] != mj.mjtJoint.mjJNT_FREE or model.nu != 10:
            raise ValueError("Expected a floating-base Roboyi with 10 actuators")
        self.joints = np.array([model.joint(name).id for name in POLICY_JOINTS])
        self.qadr = model.jnt_qposadr[self.joints]
        self.vadr = model.jnt_dofadr[self.joints]
        self.aids = np.array([model.actuator(name).id for name in POLICY_JOINTS])
        if not np.array_equal(model.actuator_trnid[self.aids, 0], self.joints):
            raise ValueError("Actuator names must refer to their matching joints")
        if not np.allclose(model.actuator_gear[self.aids, 0], 1):
            raise ValueError("Expected unit joint actuator gear")
        asset = env["scene_context"]["robot"]
        self.initial = asset["init_state"]
        self.q0 = joint_values(self.initial["joint_pos"])
        self.dq0 = joint_values(self.initial["joint_vel"])
        motor = asset["actuators"]["legs"]
        self.min_delay, self.max_delay = actuator_delay_range(motor)
        self.training_physics_dt = env["sim"]["dt"]
        self._delay_rng = np.random.default_rng(env.get("seed", 42))
        if self.max_delay:
            self._delay_step_multiplier = round(self.training_physics_dt / self.dt)
            if self._delay_step_multiplier < 1 or not math.isclose(
                self._delay_step_multiplier * self.dt, self.training_physics_dt, abs_tol=1e-10
            ):
                raise ValueError("With actuator delay, --physics-dt must also divide the training physics step "
                                 "exactly (e.g. 0.001 divides 0.005; 0.002 does not)")
        else:
            self._delay_step_multiplier = 1
        self._target_history = deque(maxlen=self.max_delay * self._delay_step_multiplier + 1)
        self.kp = joint_values(motor["stiffness"])
        self.kd = joint_values(motor["damping"])
        self.effort = joint_values(motor["effort_limit"])
        self.velocity_limit = joint_values(motor["velocity_limit_sim"])
        if not np.isfinite(self.velocity_limit).all() or np.any(self.velocity_limit <= 0):
            raise ValueError("Joint velocity limits must be finite and positive")
        self.enforce_velocity_limit = enforce_velocity_limit
        if enforce_velocity_limit and model.opt.integrator == mj.mjtIntegrator.mjINT_RK4:
            raise ValueError("Joint velocity constraints require a single-step integrator, not RK4")
        self.velocity_limit_tolerance = 1e-10
        self._limit_rhs = np.zeros((10, model.nv))
        self._limit_rhs[np.arange(10), self.vadr] = 1
        self._limit_response = np.empty_like(self._limit_rhs)
        # Match training's explicit PD once per physics step. Replace position
        # servos IN MEMORY with unit-gain motors, so there is no second PD loop.
        model.actuator_dyntype[self.aids] = mj.mjtDyn.mjDYN_NONE
        model.actuator_gaintype[self.aids] = mj.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[self.aids] = mj.mjtBias.mjBIAS_NONE
        model.actuator_gainprm[self.aids] = 0
        model.actuator_gainprm[self.aids, 0] = 1
        model.actuator_biasprm[self.aids] = 0
        model.actuator_ctrllimited[self.aids] = True
        model.actuator_forcelimited[self.aids] = True
        model.actuator_ctrlrange[self.aids] = np.column_stack((-self.effort, self.effort))
        model.actuator_forcerange[self.aids] = np.column_stack((-self.effort, self.effort))
        model.dof_armature[self.vadr] = joint_values(motor["armature"])
        model.dof_damping[self.vadr] = joint_values(motor["viscous_friction"])
        limits = model.jnt_range[self.joints]
        center = limits.mean(axis=1)
        half = np.diff(limits, axis=1).ravel() * asset["soft_joint_pos_limit_factor"] / 2
        self.lower, self.upper = center - half, center + half
        self.action_scale = env["robot"]["action_scale"]
        wrapper_clip = agent.get("clip_actions")
        self.wrapper_clip = float(wrapper_clip) if wrapper_clip is not None else math.inf
        self.env_clip = env["normalization"]["clip_actions"]
        self.obs_clip = env["normalization"]["clip_observations"]
        scales = env["normalization"]["obs_scales"]
        self.scales = np.concatenate([np.full(n, scales[key]) for key, n in
                                      (("ang_vel", 3), ("projected_gravity", 3), ("commands", 3),
                                       ("joint_pos", 10), ("joint_vel", 10), ("actions", 10))])
        self.history = np.empty((env["robot"]["actor_obs_history_length"], 39), dtype=np.float32)
        patterns = env["robot"]["terminate_contacts_body_names"] or []
        self.forbidden_bodies = {i for i in range(1, model.nbody)
                                 if any(re.fullmatch(p, model.body(i).name) for p in patterns)}
        self.reset()

    def reset(self):
        self.mj.mj_resetData(self.model, self.data)
        adr = self.model.jnt_qposadr[self.root_joint]
        self.data.qpos[adr:adr + 3] = self.initial["pos"]
        self.data.qpos[adr + 3:adr + 7] = self.initial["rot"]  # wxyz in both engines
        self.data.qpos[self.qadr] = self.q0
        self.data.qvel[self.vadr] = self.dq0
        self.previous_action = np.zeros(10, dtype=np.float32)
        self.first_frame = True
        self.target = self.q0.copy()
        # One shared lag for the legs actuator group, just as DelayedPDActuator.
        self.command_delay_steps = int(self._delay_rng.integers(self.min_delay, self.max_delay + 1))
        self.command_delay_seconds = self.command_delay_steps * self.training_physics_dt
        self._target_history.clear()
        self.applied_target = self.target.copy()
        self.last_limit_impulse = np.zeros(10)
        self.last_pre_limit_velocity = self.dq0.copy()
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.mj.mj_forward(self.model, self.data)

    def state(self):
        rotation = self.data.xmat[self.body].reshape(3, 3)
        velocity = np.empty(6)
        # XBODY selects the link frame. BODY would use inertial principal axes,
        # which differ substantially from base_link axes in this robot.
        self.mj.mj_objectVelocity(self.model, self.data, self.mj.mjtObj.mjOBJ_XBODY,
                                 self.body, velocity, 1)
        gravity = rotation.T @ np.array([0., 0., -1.])
        return velocity, gravity

    def infer(self, command):
        velocity, gravity = self.state()
        frame = np.concatenate((velocity[:3], gravity, command,
                                self.data.qpos[self.qadr] - self.q0,
                                self.data.qvel[self.vadr] - self.dq0, self.previous_action))
        frame = (frame * self.scales).astype(np.float32)
        if not np.isfinite(frame).all():
            raise FloatingPointError("Non-finite policy observation")
        if self.first_frame:
            self.history[:] = frame  # Isaac Lab CircularBuffer repeats its first append.
            self.first_frame = False
        else:
            self.history[:-1] = self.history[1:].copy()
            self.history[-1] = frame
        obs = np.clip(self.history.reshape(1, -1), -self.obs_clip, self.obs_clip)
        with torch.inference_mode():
            action = self.policy(torch.from_numpy(obs)).numpy()[0]
        if action.shape != (10,) or not np.isfinite(action).all():
            raise FloatingPointError("Non-finite or incorrectly shaped policy action")
        # Observation stores the action after the RSL-RL wrapper, before the
        # environment's own clipping/scaling (as BaseEnv.action_buffer does).
        self.previous_action = np.clip(action, -self.wrapper_clip, self.wrapper_clip)
        applied = np.clip(self.previous_action, -self.env_clip, self.env_clip)
        self.target = self.q0 + self.action_scale * applied
        if self.env["robot"]["clip_joint_targets"]:
            self.target = np.clip(self.target, self.lower, self.upper)
        return obs.copy(), self.previous_action.copy()

    def physics_step(self):
        qpos_before_step = self.data.qpos.copy() if self.enforce_velocity_limit else None
        self._target_history.append(self.target.copy())
        lag = min(self.command_delay_steps * self._delay_step_multiplier, len(self._target_history) - 1)
        # On a fresh buffer Isaac Lab repeats the oldest available command.
        self.applied_target = self._target_history[-1 - lag].copy()
        torque = self.kp * (self.applied_target - self.data.qpos[self.qadr])
        torque += self.kd * (self.dq0 - self.data.qvel[self.vadr])
        self.data.ctrl[self.aids] = np.clip(torque, -self.effort, self.effort)
        self.mj.mj_step(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("Non-finite MuJoCo state")
        self.last_pre_limit_velocity = self.data.qvel[self.vadr].copy()
        self.last_limit_impulse[:] = 0
        if self.enforce_velocity_limit:
            self.apply_velocity_limit(qpos_before_step)
        # mj_step leaves derived body transforms at the previous step. Refresh
        # them before building observations or evaluating termination.
        self.mj.mj_forward(self.model, self.data)

    def apply_velocity_limit(self, qpos_before_step):
        """Apply a mass-weighted velocity constraint impulse, then integrate pose.

        This is a split constraint solve after MuJoCo's contact/force step, not
        an independent per-joint qvel clip. Solve
            min 0.5*(v-v_free)' M (v-v_free), subject to |J v| <= limit.
        J selects joint DoFs. M^-1 J' carries the impulse response into the
        floating base and all coupled joints. No external base impulse is added.
        The dual is 0.5*p' A p + (J v_free)'p + limit'|p|,
        A = J M^-1 J'. Cyclic coordinate descent uses exact soft-threshold steps.
        """
        self.last_limit_impulse[:] = 0
        free_velocity = self.data.qvel.copy()
        joint_velocity = free_velocity[self.vadr].copy()
        if np.all(np.abs(joint_velocity) <= self.velocity_limit):
            return
        # mj_step leaves qM/factorization at the start-of-step pose, which is
        # also the pose at which its acceleration/contact forces were solved.
        self.mj.mj_solveM(self.model, self.data, self._limit_response, self._limit_rhs)
        response = self._limit_response.T
        metric = response[self.vadr]
        impulse = np.zeros(10)
        for _ in range(1000):
            for i in range(10):
                diagonal = metric[i, i]
                candidate = impulse[i] - joint_velocity[i] / diagonal
                updated = np.sign(candidate) * max(0., abs(candidate) - self.velocity_limit[i] / diagonal)
                delta = updated - impulse[i]
                impulse[i] = updated
                joint_velocity += metric[:, i] * delta
            joint_velocity = free_velocity[self.vadr] + metric @ impulse
            residual = np.where(impulse > 0, np.abs(joint_velocity + self.velocity_limit),
                                np.where(impulse < 0, np.abs(joint_velocity - self.velocity_limit),
                                         np.maximum(np.abs(joint_velocity) - self.velocity_limit, 0)))
            if np.max(residual) <= self.velocity_limit_tolerance:
                break
        else:
            raise RuntimeError("Joint velocity constraint impulse solver did not converge")
        self.data.qvel[:] = free_velocity + response @ impulse
        self.last_limit_impulse[:] = impulse
        # Re-integrate from the old pose with the constrained velocity. Otherwise
        # position would still reflect the excessive speed from the trial step.
        self.data.qpos[:] = qpos_before_step
        self.mj.mj_integratePos(self.model, self.data.qpos, self.data.qvel, self.dt)

    def failure(self):
        cfg = self.env["robot"]
        _, gravity = self.state()
        tilt = math.acos(float(np.clip(-gravity[2], -1, 1)))
        if cfg["terminate_base_height"] is not None and self.data.xpos[self.body, 2] < cfg["terminate_base_height"]:
            return "base_height"
        if cfg["terminate_base_orientation"] is not None and tilt > cfg["terminate_base_orientation"]:
            return "base_orientation"
        for index, contact in enumerate(self.data.contact):
            bodies = self.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if any(int(b) in self.forbidden_bodies for b in bodies):
                force = np.zeros(6)
                self.mj.mj_contactForce(self.model, self.data, index, force)
                if np.linalg.norm(force[:3]) > 1:
                    return "forbidden_body_contact"
        return None


def command_ranges(env, max_vx):
    """Simulation command bounds; leave the saved training configuration intact."""
    if not math.isfinite(max_vx) or max_vx <= 0:
        raise ValueError("--max-vx must be finite and positive")
    ranges = dict(env["commands"]["ranges"])
    ranges["lin_vel_x"] = (ranges["lin_vel_x"][0], max_vx)
    return ranges


class Keyboard:
    MOVEMENT_KEYS = {ord("W"): (0, .05), ord("S"): (0, -.05), ord("A"): (1, .05),
                     ord("D"): (1, -.05), ord("Q"): (2, .1), ord("E"): (2, -.1)}
    CONTROL_KEYS = frozenset(MOVEMENT_KEYS) | {32, ord("R")}

    def __init__(self, command, ranges):
        self.command = np.array(command, dtype=np.float64)
        self.bounds = np.array([ranges[key] for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")])
        self.reset_requested = False
        self.lock = threading.Lock()

    def callback(self, key):
        with self.lock:
            if key in self.MOVEMENT_KEYS:
                axis, change = self.MOVEMENT_KEYS[key]
                self.command[axis] += change
                self.command = np.clip(self.command, self.bounds[:, 0], self.bounds[:, 1])
            elif key == 32:
                self.command[:] = 0
            elif key == ord("R"):
                self.reset_requested = True
            print(f"Command vx/vy/wz: {self.command.round(3).tolist()}")

    def read(self):
        with self.lock:
            reset = self.reset_requested
            self.reset_requested = False
            return self.command.copy(), reset


class RobotViewer:
    """GLFW viewer whose robot keys never reach built-in display shortcuts.

    MuJoCo's native passive key_callback is additive: it cannot consume W/S/R
    before the native viewer also processes them. Use the project's existing
    Python viewer here so we own its window key callback. Mouse controls remain.
    """

    def __init__(self, model, data, keyboard):
        import glfw
        import mujoco_viewer

        self.glfw, self.keyboard = glfw, keyboard
        self.viewer = mujoco_viewer.MujocoViewer(
            model, data, title="Roboyi Sim2Sim", width=1280, height=800, hide_menus=True)
        self.cam = self.viewer.cam
        glfw.set_key_callback(self.viewer.window, self.key_callback)

    def key_callback(self, window, key, scancode, action, mods):
        if key in self.keyboard.CONTROL_KEYS:
            # Consume releases too: the underlying Python viewer switches many
            # rendering flags on RELEASE. Only movement keys repeat on hold.
            if action == self.glfw.PRESS or (
                    action == self.glfw.REPEAT and key in self.keyboard.MOVEMENT_KEYS):
                self.keyboard.callback(key)
            return
        if key == self.glfw.KEY_ESCAPE and action == self.glfw.PRESS:
            self.glfw.set_window_should_close(window, True)

    def is_running(self):
        return self.viewer.is_alive and not self.glfw.window_should_close(self.viewer.window)

    def sync(self):
        with self.keyboard.lock:
            vx, vy, wz = self.keyboard.command
        self.glfw.set_window_title(
            self.viewer.window,
            f"Roboyi | vx={vx:.2f} vy={vy:.2f} m/s wz={wz:.2f} rad/s | "
            "W/S speed, Q/E turn, SPACE stop, R reset, ESC exit")
        self.glfw.make_context_current(self.viewer.window)
        self.viewer.render()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.viewer.is_alive:
            self.glfw.make_context_current(self.viewer.window)
            self.viewer.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path, help="Training model_N.pt (normalizer is loaded too)")
    source.add_argument("--load-model", "--load_model", dest="load_model", type=Path, help="Exported TorchScript policy.pt")
    parser.add_argument("--config-dir", type=Path, help="Matching run's params/ (normally discovered automatically)")
    parser.add_argument("--model", type=Path, default=ROOT / "model/mujoco/standing_scene.xml")
    parser.add_argument("--export-policy", type=Path, help="Also export checkpoint actor + normalizer as TorchScript")
    parser.add_argument("--duration", type=float, default=60, help="Simulation seconds, default 60")
    parser.add_argument("--vx", type=float, default=0)
    parser.add_argument("--max-vx", type=float, default=1.0,
                        help="Forward command limit in m/s for CLI and W/S keys, default 1.0")
    parser.add_argument("--vy", type=float, default=0)
    parser.add_argument("--wz", type=float, default=0)
    parser.add_argument("--physics-dt", type=float, help="Physics step; default matches saved training config")
    parser.add_argument("--disable-joint-velocity-limit", action="store_true",
                        help="Disable the default mass-weighted joint speed constraint for comparison")
    parser.add_argument("--headless", action="store_true", help="Run without a window or OpenGL context")
    parser.add_argument("--keyboard", action="store_true", help="W/S forward, Q/E turn, space stop, R reset")
    parser.add_argument("--real-time", action="store_true", help="Pace headless runs (window runs are always paced)")
    parser.add_argument("--video", type=Path, help="Save 50-fps MP4; use MUJOCO_GL=egl on a server")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/sim2sim_roboyi" /
                        datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    if not math.isfinite(args.duration) or args.duration <= 0:
        parser.error("--duration must be finite and positive")
    if args.physics_dt is not None and (not math.isfinite(args.physics_dt) or args.physics_dt <= 0):
        parser.error("--physics-dt must be finite and positive")
    if args.keyboard and args.headless:
        parser.error("--keyboard requires a viewer window")
    if not math.isfinite(args.max_vx) or args.max_vx <= 0:
        parser.error("--max-vx must be finite and positive")
    return args


def main():
    args = parse_args()
    if args.headless and args.video:
        os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    torch.set_num_threads(1)
    source = (args.checkpoint or args.load_model).resolve()
    if not source.is_file():
        raise ValueError(f"Policy file not found: {source}")
    config_dir = args.config_dir
    if config_dir is None:
        config_dir = source.parent / "params" if args.checkpoint else source.parent.parent / "params"
    env, agent = load_config(config_dir)
    command = np.array([args.vx, args.vy, args.wz])
    ranges = command_ranges(env, args.max_vx)
    for value, key in zip(command, ("lin_vel_x", "lin_vel_y", "ang_vel_z")):
        low, high = ranges[key]
        if not np.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{key}={value} outside sim2sim command range [{low}, {high}]")
    policy, iteration = load_policy(args, env, agent)
    sim = RoboyiSim(mujoco, args.model, env, agent, policy, args.physics_dt,
                    enforce_velocity_limit=not args.disable_joint_velocity_limit)
    if args.duration < sim.control_dt:
        raise ValueError(f"--duration must be at least {sim.control_dt} seconds")
    print(f"Policy: {source}\nConfig: {config_dir}\nPolicy joint order: {POLICY_JOINTS}")
    print(f"Physics dt={sim.dt}, policy dt={sim.control_dt}, substeps={sim.substeps}")
    print(f"Actuator command delay: {sim.command_delay_steps} training physics steps "
          f"({sim.command_delay_seconds * 1000:g} ms), range {sim.min_delay}..{sim.max_delay}")
    print(f"Sim2sim vx range: {ranges['lin_vel_x']} m/s; "
          f"training vx range: {env['commands']['ranges']['lin_vel_x']} m/s")
    print("Explicit PD motors configured in memory; source XML unchanged.")
    print(f"Joint velocity constraint: {'enabled (mass-weighted impulse)' if sim.enforce_velocity_limit else 'disabled'}; "
          f"limits={sim.velocity_limit.tolist()} rad/s")
    keyboard = Keyboard(command, ranges)
    if args.keyboard:
        print("Keys: W/S vx, A/D vy (training vy is zero), Q/E wz, SPACE stop, R reset, ESC close")
    records = {key: [] for key in ("time", "episode", "command", "base_pos", "base_quat", "base_velocity",
                                   "tilt_rad", "joint_pos", "joint_vel", "joint_target", "action", "torque", "obs",
                                   "velocity_limit_impulse")}
    peak_speed = np.zeros(10)
    peak_torque = np.zeros(10)
    peak_limit_impulse = np.zeros(10)
    peak_pre_limit_speed = np.zeros(10)
    limit_active_steps = 0
    velocity_exceedances = physics_steps = resets = 0
    failure = None
    reason = "duration_reached"
    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        viewer = renderer = video = None
        if not args.headless:
            if args.keyboard:
                viewer = stack.enter_context(RobotViewer(sim.model, sim.data, keyboard))
            else:
                import mujoco.viewer
                viewer = stack.enter_context(mujoco.viewer.launch_passive(sim.model, sim.data))
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.6, 135, -20
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            renderer = stack.enter_context(mujoco.Renderer(sim.model, height=720, width=1280))
            video = stack.enter_context(imageio.get_writer(str(args.video), fps=1 / sim.control_dt))
            camera = mujoco.MjvCamera()
            camera.distance, camera.azimuth, camera.elevation = 1.6, 135, -20
        try:
            for step in range(int(args.duration / sim.control_dt)):
                tick = time.monotonic()
                if viewer is not None and not viewer.is_running():
                    reason = "viewer_closed"
                    break
                command, reset = keyboard.read()
                if reset:
                    sim.reset()
                    resets += 1
                obs, action = sim.infer(command)
                velocity, gravity = sim.state()
                values = ((step * sim.control_dt), resets, command, sim.data.xpos[sim.body].copy(),
                          sim.data.xquat[sim.body].copy(), velocity.copy(),
                          math.acos(float(np.clip(-gravity[2], -1, 1))), sim.data.qpos[sim.qadr].copy(),
                          sim.data.qvel[sim.vadr].copy(), sim.target.copy(), action, None, obs[0])
                for key, value in zip(list(records)[:-1], values):
                    if key != "torque":
                        records[key].append(value)
                for _ in range(sim.substeps):
                    sim.physics_step()
                    physics_steps += 1
                    speed = np.abs(sim.data.qvel[sim.vadr])
                    torque = np.abs(sim.data.actuator_force[sim.aids])
                    peak_speed = np.maximum(peak_speed, speed)
                    peak_torque = np.maximum(peak_torque, torque)
                    peak_pre_limit_speed = np.maximum(peak_pre_limit_speed, np.abs(sim.last_pre_limit_velocity))
                    peak_limit_impulse = np.maximum(peak_limit_impulse, np.abs(sim.last_limit_impulse))
                    limit_active_steps += int(np.any(sim.last_limit_impulse))
                    velocity_exceedances += int(np.any(speed > sim.velocity_limit + 1e-6))
                    failure = sim.failure()
                    if failure or np.any(sim.data.warning.number):
                        failure = failure or "mujoco_warning"
                        break
                records["torque"].append(sim.data.actuator_force[sim.aids].copy())
                records["velocity_limit_impulse"].append(sim.last_limit_impulse.copy())
                if viewer is not None:
                    viewer.cam.lookat[:] = sim.data.xpos[sim.body]
                    viewer.sync()
                if video is not None:
                    camera.lookat[:] = sim.data.xpos[sim.body]
                    renderer.update_scene(sim.data, camera=camera)
                    video.append_data(renderer.render())
                if failure:
                    reason = failure
                    break
                if viewer is not None or args.real_time:
                    time.sleep(max(0, sim.control_dt - (time.monotonic() - tick)))
        except KeyboardInterrupt:
            reason = "interrupted"
    arrays = {key: np.asarray(value) for key, value in records.items()}
    np.savez_compressed(args.output_dir / "trajectory.npz", **arrays, joint_names=np.array(POLICY_JOINTS))
    summary = {
        "policy": str(source), "checkpoint_iteration": iteration, "config_dir": str(config_dir.resolve()),
        "model": str(args.model.resolve()), "joint_names": POLICY_JOINTS,
        "physics_dt": sim.dt, "control_dt": sim.control_dt, "physics_steps": physics_steps,
        "actuator_delay_range_training_steps": [sim.min_delay, sim.max_delay],
        "actuator_delay_training_steps": sim.command_delay_steps,
        "actuator_delay_seconds": sim.command_delay_seconds,
        "command_ranges": {key: list(ranges[key]) for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "training_command_ranges": {key: list(env["commands"]["ranges"][key])
                                    for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z")},
        "simulated_seconds": physics_steps * sim.dt, "requested_seconds": args.duration,
        "wall_seconds": time.monotonic() - started, "stop_reason": reason,
        "fell": reason in ("base_height", "base_orientation", "forbidden_body_contact"),
        "resets": resets, "peak_joint_speed_rad_s": peak_speed.tolist(),
        "peak_actuator_torque_Nm": peak_torque.tolist(),
        "training_velocity_limit_rad_s": sim.velocity_limit.tolist(),
        "velocity_limit_exceedance_step_fraction": velocity_exceedances / max(1, physics_steps),
        "velocity_limit_enforced": sim.enforce_velocity_limit,
        "velocity_limit_method": "mass_weighted_split_impulse" if sim.enforce_velocity_limit else "disabled",
        "velocity_limit_solver_tolerance_rad_s": sim.velocity_limit_tolerance,
        "velocity_limit_active_steps": limit_active_steps,
        "peak_pre_limit_joint_speed_rad_s": peak_pre_limit_speed.tolist(),
        "peak_velocity_limit_impulse_Nm_s": peak_limit_impulse.tolist(),
        "mujoco_warning_counts": sim.data.warning.number.tolist(),
        "final_base_position": sim.data.xpos[sim.body].tolist(),
        "max_sampled_tilt_deg": float(np.rad2deg(arrays["tilt_rad"]).max()) if len(arrays["time"]) else None,
        "mean_body_velocity_after_warmup": arrays["base_velocity"][len(arrays["time"]) // 5:].mean(axis=0).tolist()
        if len(arrays["time"]) else None,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Saved results: {args.output_dir}")
    return 2 if failure else 0


if __name__ == "__main__":
    sys.exit(main())
