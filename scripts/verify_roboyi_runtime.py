"""Exercise the real GPU environment, checkpoint inference, limits and resets.

Run with scripts/python.sh and pass the PPO checkpoint produced by train.py.
This is an integration check; it does not assess walking quality.
"""

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=128)
parser.add_argument("--output", type=Path, default=Path("outputs/roboyi_ppo_verification/runtime.json"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from tensordict import TensorDict
from rsl_rl.modules import ActorCritic
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import robolab.tasks  # noqa: F401


def main():
    cfg = load_cfg_from_registry("Roboyi-Flat", "env_cfg_entry_point")
    cfg.scene.num_envs = args.num_envs
    cfg.seed = 42
    if args.device is not None:
        cfg.sim.device = args.device
    env = gym.make("Roboyi-Flat", cfg=cfg)
    try:
        base = env.unwrapped
        robot = base.robot
        obs, _ = env.reset()
        assert obs["policy"].shape == (args.num_envs, cfg.observation_space) == (args.num_envs, 390)
        assert obs["critic"].shape == (args.num_envs, cfg.state_space) == (args.num_envs, 740)
        assert robot.num_instances == args.num_envs and robot.num_joints == 10
        assert not robot.is_fixed_base
        assert set(robot.joint_names) == set(robot.cfg.init_state.joint_pos)
        assert len(base.feet_cfg.body_ids) == 2
        actuator = robot.actuators["legs"]
        delay_steps = actuator.positions_delay_buffer.time_lags.clone()
        assert (delay_steps >= actuator.cfg.min_delay).all()
        assert (delay_steps <= actuator.cfg.max_delay).all()
        torch.testing.assert_close(delay_steps, actuator.velocities_delay_buffer.time_lags)
        torch.testing.assert_close(delay_steps, actuator.efforts_delay_buffer.time_lags)
        view = robot.root_physx_view
        # Explicit PD must replace USD drives, rather than apply a second PD loop.
        assert torch.count_nonzero(view.get_dof_stiffnesses()) == 0
        assert torch.count_nonzero(view.get_dof_dampings()) == 0
        torch.testing.assert_close(view.get_dof_max_velocities(), torch.full_like(view.get_dof_max_velocities(), 3.0))
        torch.testing.assert_close(view.get_dof_max_forces(), torch.full_like(view.get_dof_max_forces(), 20.0))
        viscous = view.get_dof_friction_properties()[..., 2]
        torch.testing.assert_close(viscous, torch.full_like(viscous, 0.01))
        assert (view.get_masses() > 0).all()
        assert "joint_deviation_torso" not in base.reward_manager.active_terms
        assert "joint_deviation_arms" not in base.reward_manager.active_terms
        base._pre_physics_step(torch.full((args.num_envs, 10), 1000.0, device=base.device))
        limits = robot.data.soft_joint_pos_limits
        assert (base.actions <= limits[..., 1] + 1e-6).all()
        assert (base.actions >= limits[..., 0] - 1e-6).all()
        obs, _ = env.reset()

        agent_cfg = load_cfg_from_registry("Roboyi-Flat", "rsl_rl_cfg_entry_point")
        policy_cfg = agent_cfg.policy.to_dict()
        policy_cfg.pop("class_name")
        policy = ActorCritic(TensorDict(obs, batch_size=[args.num_envs]), agent_cfg.obs_groups, 10, **policy_cfg).to(base.device)
        checkpoint = torch.load(args.checkpoint, map_location=base.device, weights_only=False)
        policy.load_state_dict(checkpoint["model_state_dict"])
        policy.eval()
        reset_count = 0
        # Force a real timeout through step() to check the normal reset path.
        base.episode_length_buf[0] = int(base.episode_length) - 1
        for i in range(args.steps):
            with torch.inference_mode():
                actions = policy.act_inference(TensorDict(obs, batch_size=[args.num_envs]))
                assert actions.shape == (args.num_envs, 10) and torch.isfinite(actions).all()
                obs, reward, terminated, truncated, _ = env.step(actions)
            if i == 0:
                assert bool(truncated[0]), "The forced timeout did not trigger a reset"
            reset_count += int((terminated | truncated).sum())
            assert all(torch.isfinite(value).all() for value in obs.values())
            assert torch.isfinite(reward).all()
            assert torch.isfinite(robot.data.root_state_w).all()
            assert torch.count_nonzero(base.command_generator.command[:, 1]) == 0
            assert (base.actions <= limits[..., 1] + 1e-6).all()
            assert (base.actions >= limits[..., 0] - 1e-6).all()
        result = {
            "passed": True,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_iteration": checkpoint["iter"],
            "device": str(base.device),
            "num_envs": args.num_envs,
            "steps": args.steps,
            "resets": reset_count,
            "joints": robot.joint_names,
            "body_count": robot.num_bodies,
            "actor_shape": list(obs["policy"].shape),
            "critic_shape": list(obs["critic"].shape),
            "active_rewards": base.reward_manager.active_terms,
            "velocity_limit_rad_s": view.get_dof_max_velocities()[0].cpu().tolist(),
            "effort_limit_Nm": view.get_dof_max_forces()[0].cpu().tolist(),
            "viscous_friction": viscous[0].cpu().tolist(),
            "actuator_delay_range_steps": [actuator.cfg.min_delay, actuator.cfg.max_delay],
            "sampled_actuator_delay_steps": delay_steps.cpu().tolist(),
            "physics_dt": base.physics_dt,
            "checks": ["floating_base", "all_10_actuators", "finite_policy_inference", "finite_observations_rewards",
                       "timeout_reset", "joint_target_limits", "no_lateral_commands", "no_double_pd", "passive_damping",
                       "actuator_delay_bounds_and_group_consistency"],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        print("ROBOYI_RUNTIME_VERIFIED", json.dumps(result), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
