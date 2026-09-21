from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from . import mdp

##
# Pre-defined configs
##

from isaaclab_assets import G1_MINIMAL_CFG  # isort: skip


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain: flat, so the pelvis height can be compared to an absolute target and the
    # robot only has to solve the recovery, not the terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/torso_link",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


##
# MDP settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # the 14 finger joints are useless to stand up and only dilute the exploration noise,
    # so only the 23 body joints are actuated by the policy
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*", "torso_joint", ".*_shoulder_.*", ".*_elbow_.*"],
        scale=0.5,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )

    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )

    # reset
    # start from a state a real fall produces: the bank is recorded offline by dropping robots
    # and letting them settle, so the episode begins with the robot already at rest on the ground
    # instead of in mid-air. Generate it with 'python scripts/make_fallen_bank.py'.
    reset_from_bank = EventTerm(
        func=mdp.reset_from_fallen_bank,
        mode="reset",
    )

    # HoST-style assist: hold 80% of the robot weight up at the start of training and fade it to
    # zero over the first 30k steps, so the policy first discovers what standing feels like
    upward_assist = EventTerm(
        func=mdp.apply_upward_assist,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "max_force_scale": 0.8,
            "decay_steps": 30000,
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task: the product carries the objective, so height and uprightness cannot be cashed in
    # separately. No sensor_cfg on flat ground, the target is absolute.
    stand_up_exp = RewTerm(
        func=mdp.stand_up_exp,
        weight=2.0,
        params={
            "target_height": 0.74,
            "height_std": 0.5,
            "upright_std": 1.0,
        },
    )
    # -- shaping, kept at low weight mostly so both halves stay visible in the logs
    track_height = RewTerm(
        func=mdp.track_height,
        weight=0.25,
        params={
            "target_height": 0.74,
            "std": 0.5,
        },
    )
    upright_exp = RewTerm(func=mdp.upright_exp, weight=0.25, params={"std": 1.0})
    # -- posture: these fight the get-up motion itself, which needs to slide and swing, so they
    # all start at zero and CurriculumCfg switches them on once standing is already learned
    base_lin_vel_xy_l2 = RewTerm(func=mdp.base_lin_vel_xy_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=0.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_ankle_roll_link"),
        },
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_shoulder_.*", ".*_elbow_.*"])},
    )
    # -- penalties
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.002)
    # -- optional penalties
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_ankle_.*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # only the time out: on a fall recovery task the robot starts on the ground, so terminating
    # on torso contact would end every episode at the first step
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # 40k steps is well past the end of the upward assist (30k), so the policy already stands up
    # on its own before it is asked to do it tidily
    base_lin_vel_xy_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "base_lin_vel_xy_l2", "weight": -0.3, "num_steps": 40000},
    )
    ang_vel_xy_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "ang_vel_xy_l2", "weight": -0.05, "num_steps": 40000},
    )
    feet_slide = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "feet_slide", "weight": -0.1, "num_steps": 40000},
    )
    joint_deviation_arms = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "joint_deviation_arms", "weight": -0.05, "num_steps": 40000},
    )
    action_rate_l2 = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "action_rate_l2", "weight": -0.005, "num_steps": 40000},
    )
    dof_pos_limits = CurrTerm(
        func=mdp.modify_reward_weight,
        params={"term_name": "dof_pos_limits", "weight": -1.0, "num_steps": 40000},
    )


##
# Environment configuration
##


@configclass
class G1FallRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the G1 fall recovery environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # scene settings
        self.scene.robot = G1_MINIMAL_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # flat ground: nothing reads the height scanner any more
        self.scene.height_scanner = None
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # no terrain curriculum: it promotes envs based on the distance walked against a velocity
        # command, which does not exist on a fall recovery task
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = False
