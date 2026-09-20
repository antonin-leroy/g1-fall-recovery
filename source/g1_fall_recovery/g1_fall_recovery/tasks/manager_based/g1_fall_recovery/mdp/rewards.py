# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import wrap_to_pi

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def joint_pos_target_l2(env: ManagerBasedRLEnv, target: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize joint position deviation from a target value."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # wrap the joint positions to (-pi, pi)
    joint_pos = wrap_to_pi(asset.data.joint_pos[:, asset_cfg.joint_ids])
    # compute the reward
    return torch.sum(torch.square(joint_pos - target), dim=1)


def track_height(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward the root height for staying close to a target, using an exponential kernel.

    Note:
        On flat terrain the target is in the world frame. On rough terrain, pass a ray-caster
        sensor so the target is offset by the measured ground height under the robot.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # adjust the target to the ground height when a height scanner is available
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        adjusted_target_height = target_height
    # compute the error
    height_error = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    return torch.exp(-height_error / std**2)


def upright_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward an upright base using an exponential kernel on the projected gravity vector.

    Note:
        ``projected_gravity_b[:, 2]`` is -1 when upright, 0 on the side and +1 upside down.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the error against the upright reference (0, 0, -1)
    tilt_error = torch.square(asset.data.projected_gravity_b[:, 2] + 1.0)
    return torch.exp(-tilt_error / std**2)
