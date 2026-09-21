# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to record a bank of settled fallen states for the fall recovery task.

Robots are dropped from a height with a fully random orientation and a random joint target held
constant, then left to settle under gravity. The resulting poses are what a real fall leaves
behind, which is what the policy should start from.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Record a bank of settled fallen states.")
parser.add_argument("--task", type=str, default="Template-G1-Fall-Recovery-v0", help="Name of the task.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to drop at once.")
parser.add_argument("--num_rounds", type=int, default=8, help="Number of drop rounds to record.")
parser.add_argument("--settle_steps", type=int, default=250, help="Environment steps to let the robots settle.")
parser.add_argument("--drop_height", type=float, default=0.9, help="Pelvis height at the start of a drop, in meters.")
parser.add_argument("--output", type=str, default=None, help="Where to write the bank. Defaults to the mdp folder.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os

import gymnasium as gym
import torch

from isaaclab.utils.math import random_orientation

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg

import g1_fall_recovery.tasks  # noqa: F401
from g1_fall_recovery.tasks.manager_based.g1_fall_recovery.mdp.events import DEFAULT_BANK_PATH


def main():
    """Drop robots, let them settle, and store the poses they end up in."""
    output = args_cli.output or DEFAULT_BANK_PATH

    # parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    # the bank does not exist yet, so the event that reads it has to be off while we record it
    env_cfg.events.reset_from_bank = None
    # and the training assist must be off too, otherwise the robots are held up as they fall
    env_cfg.events.upward_assist = None
    # the episode must outlast a settling round, otherwise a time out would reset the robots mid-drop
    env_cfg.episode_length_s = 1e6
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    robot = env.scene["robot"]
    device = env.device
    num_envs = env.num_envs
    action_dim = env.action_manager.total_action_dim

    env.reset()

    root_poses = []
    joint_positions = []

    for round_idx in range(args_cli.num_rounds):
        # drop the robots from a height, each with a different random orientation
        root_pose = torch.zeros(num_envs, 7, device=device)
        root_pose[:, 0:3] = env.scene.env_origins
        root_pose[:, 2] += args_cli.drop_height
        root_pose[:, 3:7] = random_orientation(num_envs, device=device)
        robot.write_root_pose_to_sim(root_pose)
        robot.write_root_velocity_to_sim(torch.zeros(num_envs, 6, device=device))

        # scatter the joints around the default pose so the robots land in varied heaps
        joint_pos = robot.data.default_joint_pos * torch.empty(
            num_envs, robot.num_joints, device=device
        ).uniform_(0.4, 1.6)
        joint_pos = torch.clamp(
            joint_pos, robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
        )
        robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos))

        # hold one random posture per robot while it falls, so it does not land as a rigid statue
        actions = torch.empty(num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
        for _ in range(args_cli.settle_steps):
            env.step(actions)

        # settling does not always resolve interpenetration, and a state written back with a limb
        # inside the floor makes the episode start on a depenetration kick. Lift each robot until
        # its lowest body clears the ground.
        body_z = robot.data.body_pos_w[..., 2] - env.scene.env_origins[:, 2].unsqueeze(1)
        lowest = body_z.min(dim=1).values
        lift = torch.clamp(-lowest, min=0.0) + 0.005

        # record the settled state, with positions relative to the environment origin
        root_pose = robot.data.root_state_w[:, 0:7].clone()
        root_pose[:, 0:3] -= env.scene.env_origins
        root_pose[:, 2] += lift
        root_poses.append(root_pose.cpu())
        joint_positions.append(robot.data.joint_pos.clone().cpu())
        print(
            f"[INFO]: round {round_idx + 1}/{args_cli.num_rounds} recorded ({num_envs} states),"
            f" {(lowest < 0).float().mean() * 100:.0f}% were clipping the floor,"
            f" max lift {lift.max():.3f} m"
        )

    bank = {"root_pose": torch.cat(root_poses), "joint_pos": torch.cat(joint_positions)}
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(bank, output)

    heights = bank["root_pose"][:, 2]
    print(f"[INFO]: wrote {bank['root_pose'].shape[0]} fallen states to {output}")
    print(f"[INFO]: pelvis height min {heights.min():.3f} m, mean {heights.mean():.3f} m, max {heights.max():.3f} m")

    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
