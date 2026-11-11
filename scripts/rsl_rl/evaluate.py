# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate a trained get-up policy over many fallen states, headless, and save the raw stats.

Every env runs one full episode (10 s) from a state of the fallen bank. The script saves, per env:
the initial orientation, the pelvis height and uprightness at every step, and the peak joint torque
and velocity as a fraction of the motor limits. It also saves the full poses of the first envs,
for a MuJoCo replay. ``--push`` adds a random horizontal velocity kick to the pelvis at t = 5 s,
once the robot is standing, to see whether it falls and whether it gets up again.

The analysis and the plots are done offline from the .npz (``scripts/analyze_eval.py``).

Example (on the pod):

    python scripts/rsl_rl/evaluate.py --task Template-G1-Fall-Recovery-v0 --headless --num_envs 4096 \
        --checkpoint logs/rsl_rl/g1_fall_recovery/<run>/model_2999.pt --output eval_nominal.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a get-up policy.")
parser.add_argument("--num_envs", type=int, default=4096, help="Number of fallen states to evaluate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--push", type=float, default=0.0, help="Max horizontal kick (m/s) at t = 5 s, 0 = no push.")
parser.add_argument("--push_time", type=float, default=5.0, help="When the kick happens (s).")
parser.add_argument("--record_envs", type=int, default=64, help="Envs whose full poses are saved for replay.")
parser.add_argument("--export", action="store_true", help="Also export the policy to TorchScript and ONNX.")
parser.add_argument("--output", type=str, required=True, help="Output .npz path.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import re

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

import isaaclab.envs.mdp as base_mdp
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import g1_fall_recovery.tasks  # noqa: F401

FINGER_RE = re.compile(r"_(zero|one|two|three|four|five|six)_joint$")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # same playback settings as play.py: no training assist, clean observations
    if getattr(env_cfg.events, "upward_assist", None) is not None:
        env_cfg.events.upward_assist = None
    env_cfg.observations.policy.enable_corruption = False

    if args_cli.push > 0.0:
        # one kick per env at push_time: the interval timer restarts at every reset and the
        # episode (10 s) ends before a second one could fire
        v = args_cli.push
        env_cfg.events.eval_push = EventTermCfg(
            func=base_mdp.push_by_setting_velocity,
            mode="interval",
            interval_range_s=(args_cli.push_time, args_cli.push_time),
            params={"velocity_range": {"x": (-v, v), "y": (-v, v)}},
        )
        print(f"[INFO]: push of up to {v} m/s per axis at t = {args_cli.push_time} s")

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    env_cfg.log_dir = os.path.dirname(resume_path)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    if args_cli.export:
        policy_nn = runner.alg.policy
        normalizer = getattr(policy_nn, "actor_obs_normalizer", None)
        export_dir = os.path.join(os.path.dirname(resume_path), "exported")
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.pt")
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_dir, filename="policy.onnx")
        print(f"[INFO]: policy exported to {export_dir}")

    robot = env.unwrapped.scene["robot"]
    origins = env.unwrapped.scene.env_origins
    n = env.num_envs
    k = min(args_cli.record_envs, n)
    dt = env.unwrapped.step_dt
    num_steps = int(round(env.unwrapped.max_episode_length))  # 500: the last step is the time-out reset

    body = torch.tensor([not FINGER_RE.search(name) for name in robot.joint_names], device=robot.device)
    effort_limit = robot.data.joint_effort_limits.clone()  # (n, J), the sim limits of G1_REAL_CFG
    vel_limit = robot.data.joint_vel_limits.clone()

    height = torch.zeros(num_steps, n, device=robot.device)
    gz = torch.zeros(num_steps, n, device=robot.device)
    speed_xy = torch.zeros(num_steps, n, device=robot.device)
    peak_torque = torch.zeros(n, robot.num_joints, device=robot.device)
    peak_vel = torch.zeros(n, robot.num_joints, device=robot.device)
    rec_pos = np.zeros((num_steps, k, 3), np.float32)
    rec_quat = np.zeros((num_steps, k, 4), np.float32)
    rec_joint = np.zeros((num_steps, k, robot.num_joints), np.float32)
    pos_xy = torch.zeros(num_steps, n, 2, device=robot.device)

    # initial fallen state, before the first step
    init_gravity = robot.data.projected_gravity_b.clone()
    init_height = robot.data.root_pos_w[:, 2].clone()

    obs = env.get_observations()
    for t in range(num_steps):
        root_pos = robot.data.root_pos_w - origins
        height[t] = root_pos[:, 2]
        pos_xy[t] = root_pos[:, :2]
        gz[t] = robot.data.projected_gravity_b[:, 2]
        speed_xy[t] = torch.linalg.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
        rec_pos[t] = root_pos[:k].cpu().numpy()
        rec_quat[t] = robot.data.root_quat_w[:k].cpu().numpy()
        rec_joint[t] = robot.data.joint_pos[:k].cpu().numpy()
        if t == num_steps - 1:
            break  # the next step is the time-out, which resets every env
        with torch.inference_mode():
            obs, _, dones, _ = env.step(policy(obs))
        # applied_torque is the PD torque after clipping, as computed for the implicit actuators
        peak_torque = torch.maximum(peak_torque, robot.data.applied_torque.abs() / effort_limit)
        peak_vel = torch.maximum(peak_vel, robot.data.joint_vel.abs() / vel_limit)
        if dones.any():
            print(f"[WARN]: {int(dones.sum())} envs reset early at step {t}")
        if t % 100 == 0:
            print(f"[INFO]: step {t}/{num_steps}")

    np.savez_compressed(
        args_cli.output,
        dt=np.float32(dt),
        push=np.float32(args_cli.push),
        push_time=np.float32(args_cli.push_time),
        joint_names=np.array(robot.joint_names),
        body_joint=body.cpu().numpy(),
        init_gravity=init_gravity.cpu().numpy(),
        init_height=init_height.cpu().numpy(),
        height=height.cpu().numpy().astype(np.float16),  # (T, n)
        gz=gz.cpu().numpy().astype(np.float16),  # (T, n)
        speed_xy=speed_xy.cpu().numpy().astype(np.float16),  # (T, n)
        pos_xy=pos_xy.cpu().numpy().astype(np.float16),  # (T, n, 2)
        peak_torque=peak_torque.cpu().numpy(),  # (n, J), fraction of the limit
        peak_vel=peak_vel.cpu().numpy(),  # (n, J), fraction of the limit
        effort_limit=effort_limit[0].cpu().numpy(),
        vel_limit=vel_limit[0].cpu().numpy(),
        # same keys as record_poses.py, so replay_mujoco.py reads this file directly
        root_pos=rec_pos,
        root_quat=rec_quat,
        joint_pos=rec_joint,
        dones=np.zeros((num_steps, k), bool),
    )
    print(f"[INFO]: saved {num_steps} steps x {n} envs to {args_cli.output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
