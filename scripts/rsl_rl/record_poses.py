# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained RSL-RL checkpoint headless and save the robot's poses to a .npz file.

The RTX renderer of Isaac Sim 5.1 crashes on the R595 drivers Runpod hosts now run, so this
script records no video. Instead it logs, at every policy step, the root pose and the joint
angles of the first few environments. ``scripts/replay_mujoco.py`` then draws those poses in
MuJoCo on any machine (no physics, a pure kinematic replay) and writes the mp4.

Example (on the pod):

    python scripts/rsl_rl/record_poses.py --task Template-G1-Fall-Recovery-v0 --headless \
        --num_envs 8 --checkpoint logs/rsl_rl/<experiment>/<run>/model_2999.pt
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Record robot poses from a trained RSL-RL policy.")
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--num_steps", type=int, default=500, help="Policy steps to record (500 = 10 s at 50 Hz).")
parser.add_argument("--record_envs", type=int, default=4, help="How many environments (0, 1, ...) to record.")
parser.add_argument("--output", type=str, default=None, help="Output .npz path (default: <run dir>/poses.npz).")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import g1_fall_recovery.tasks  # noqa: F401


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Run the policy and log root pose + joint angles."""
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs

    # same playback settings as play.py: no training assist, clean observations
    if getattr(env_cfg.events, "upward_assist", None) is not None:
        env_cfg.events.upward_assist = None
        print("[INFO]: upward assist disabled for playback")
    env_cfg.observations.policy.enable_corruption = False

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # find the checkpoint
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    output_path = args_cli.output or os.path.join(log_dir, "poses.npz")

    # create the environment (no render_mode: nothing touches the RTX renderer)
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    # each env is spawned at its own spot on a grid: subtract it to bring every robot back to (0, 0)
    env_origins = env.unwrapped.scene.env_origins
    n = min(args_cli.record_envs, env.num_envs)

    def snapshot():
        root_pos = (robot.data.root_pos_w[:n] - env_origins[:n]).cpu().numpy()
        root_quat = robot.data.root_quat_w[:n].cpu().numpy()  # (w, x, y, z), the same order as MuJoCo
        joint_pos = robot.data.joint_pos[:n].cpu().numpy()
        return root_pos, root_quat, joint_pos

    root_pos_log, root_quat_log, joint_pos_log, done_log = [], [], [], []

    # frame 0: the fallen pose the episode starts from
    obs = env.get_observations()
    p, q, j = snapshot()
    root_pos_log.append(p)
    root_quat_log.append(q)
    joint_pos_log.append(j)
    done_log.append(np.zeros(n, dtype=bool))

    for step in range(args_cli.num_steps):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
        # after a done, Isaac Lab has already reset that env: this frame is the NEXT episode's start
        p, q, j = snapshot()
        root_pos_log.append(p)
        root_quat_log.append(q)
        joint_pos_log.append(j)
        done_log.append(dones[:n].cpu().numpy().astype(bool))
        if step % 100 == 0:
            print(f"[INFO]: step {step}/{args_cli.num_steps}")

    np.savez_compressed(
        output_path,
        root_pos=np.stack(root_pos_log).astype(np.float32),  # (T, n, 3)
        root_quat=np.stack(root_quat_log).astype(np.float32),  # (T, n, 4)
        joint_pos=np.stack(joint_pos_log).astype(np.float32),  # (T, n, num_joints)
        dones=np.stack(done_log),  # (T, n)
        joint_names=np.array(robot.joint_names),
        dt=np.float32(env.unwrapped.step_dt),
    )
    print(f"[INFO]: saved {len(root_pos_log)} frames x {n} envs to {output_path}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
