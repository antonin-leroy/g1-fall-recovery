# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unitree G1 articulation with the collision model and motor limits of the real robot.

The Isaac Lab ``G1_MINIMAL_CFG`` is built for walking: only the feet and the torso collide, self
collision is off, every hip, knee, shoulder and elbow gets 300 Nm and no joint has a speed limit.
On a get-up task that lets the legs, pelvis and arms sink through the floor and the arms throw the
whole robot up. The limits below come from Unitree's robot description (unitree_ros,
``g1_description``) and the gains and motor inertias from Unitree's Isaac Lab configs
(unitree_rl_lab).
"""

import math

from isaaclab.actuators import ImplicitActuatorCfg

from isaaclab_assets import G1_CFG  # isort: skip

# reflected inertia of each Unitree motor at the joint: rotor and gear stages times the gear ratio squared
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925

# PD gains from a 10 Hz natural frequency and a damping ratio of 2 on the motor inertia, as in unitree_rl_lab
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


G1_REAL_CFG = G1_CFG.replace(
    # the full model: colliders on 38 of the 44 bodies, where the minimal one only has the feet and torso
    spawn=G1_CFG.spawn.replace(
        # getting up is all limbs folding against the body, so they must not pass through each other
        articulation_props=G1_CFG.spawn.articulation_props.replace(enabled_self_collisions=True),
    ),
    actuators={
        # N7520-14.3 motors
        "hip_pitch_yaw_torso": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch_joint", ".*_hip_yaw_joint", "torso_joint"],
            effort_limit_sim=88.0,
            velocity_limit_sim=32.0,
            stiffness=_stiffness(ARMATURE_7520_14),
            damping=_damping(ARMATURE_7520_14),
            armature=ARMATURE_7520_14,
        ),
        # N7520-22.5 motors
        "hip_roll_knee": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_roll_joint", ".*_knee_joint"],
            effort_limit_sim=139.0,
            velocity_limit_sim=20.0,
            stiffness=_stiffness(ARMATURE_7520_22),
            damping=_damping(ARMATURE_7520_22),
            armature=ARMATURE_7520_22,
        ),
        # two N5020 motors drive each ankle through a parallel linkage, hence the doubled inertia
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=35.0,
            velocity_limit_sim=30.0,
            stiffness=_stiffness(2.0 * ARMATURE_5020),
            damping=_damping(2.0 * ARMATURE_5020),
            armature=2.0 * ARMATURE_5020,
        ),
        # N5020 motors; the elbow roll of this older model sits where the wrist roll is on the current G1
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_shoulder_.*", ".*_elbow_.*"],
            effort_limit_sim=25.0,
            velocity_limit_sim=37.0,
            stiffness=_stiffness(ARMATURE_5020),
            damping=_damping(ARMATURE_5020),
            armature=ARMATURE_5020,
        ),
        # Dex3 fingers: not driven by the policy, they only hold the default grasp
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[".*_zero_joint", ".*_one_joint", ".*_two_joint", ".*_three_joint",
                              ".*_four_joint", ".*_five_joint", ".*_six_joint"],
            effort_limit_sim=1.4,
            velocity_limit_sim=12.0,
            stiffness=40.0,
            damping=10.0,
            armature=0.001,
        ),
    },
)
"""Unitree G1 (37 DOF, Dex3 hands) with full collisions, self collision and real motor limits."""

G1_ACTION_SCALE: dict[str, float] = {
    expr: 0.25 * actuator.effort_limit_sim / actuator.stiffness
    for name, actuator in G1_REAL_CFG.actuators.items()
    if name != "hands"
    for expr in actuator.joint_names_expr
}
"""Per-joint action scale, as in unitree_rl_lab: a unit action asks for a quarter of the torque limit."""
