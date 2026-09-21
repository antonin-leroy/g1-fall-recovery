# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

DEFAULT_BANK_PATH = os.path.join(os.path.dirname(__file__), "fallen_states.pt")
"""Where ``scripts/make_fallen_bank.py`` writes the bank by default."""

_BANK_CACHE: dict[str, dict[str, torch.Tensor]] = {}


def _load_bank(path: str, device: str) -> dict[str, torch.Tensor]:
    """Load the bank once per path and keep it on the simulation device."""
    key = f"{path}:{device}"
    if key not in _BANK_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"No fallen state bank at '{path}'. Generate one first with"
                " 'python scripts/make_fallen_bank.py'."
            )
        bank = torch.load(path, map_location=device)
        _BANK_CACHE[key] = {k: v.to(device) for k, v in bank.items()}
    return _BANK_CACHE[key]


def reset_from_fallen_bank(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    bank_path: str = DEFAULT_BANK_PATH,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset to a pose sampled from a bank of settled fallen states.

    The bank is recorded offline by dropping robots and letting them settle, so every starting
    state is one physics actually produces. The episode then begins with the robot already at
    rest on the ground rather than in mid-air.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    bank = _load_bank(bank_path, env.device)
    # sample one recorded state per environment being reset
    indices = torch.randint(0, bank["root_pose"].shape[0], (len(env_ids),), device=env.device)
    # the bank stores positions relative to the environment origin
    root_pose = bank["root_pose"][indices].clone()
    root_pose[:, 0:3] += env.scene.env_origins[env_ids]
    asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    # the robot is at rest, so the joint velocities start at zero
    joint_pos = bank["joint_pos"][indices].clone()
    asset.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
