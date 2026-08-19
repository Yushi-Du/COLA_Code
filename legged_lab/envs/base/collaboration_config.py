"""Shared configuration schemas for phase-2 teachers and phase-3 students."""

from dataclasses import MISSING
import math

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.envs.base import collaboration_mdp
from legged_lab.envs.base.common_config import (
    BarControllerCfg,
    CollaborationExperimentCfg,
    CollaborationGraspCfg,
    CollaborationObservationCfg,
    SupportHeightControlCfg,
    SupportMotionCfg,
    SupportResetCfg,
    TerminationObservationCfg,
)
from legged_lab.utils.rsl_rl_cfg import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationStudentTeacherCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticResidualCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class CollaborationRewardBaseCfg:
    pass

@configclass
class HeightScannerCfg:
    enable_height_scan: bool = False
    prim_body_name: str = MISSING
    resolution: float = 0.1
    size: tuple = (1.6, 1.0)
    debug_vis: bool = False
    drift_range: tuple = (0.0, 0.0)


@configclass
class CollaborationSceneSettingsCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class TeacherRobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []


@configclass
class StudentRobotCfg:
    teacher_obs_history_length: int = 10
    student_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []


def teacher_robot_cfg() -> TeacherRobotCfg:
    return TeacherRobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=MISSING,
        feet_body_names=MISSING,
    )


def student_robot_cfg() -> StudentRobotCfg:
    return StudentRobotCfg(
        teacher_obs_history_length=10,
        student_obs_history_length=25,
        action_scale=0.25,
        terminate_contacts_body_names=MISSING,
        feet_body_names=MISSING,
    )


@configclass
class ObsScalesCfg:
    lin_vel: float = 1.0
    ang_vel: float = 0.25
    projected_gravity: float = 1.0
    commands: float = 1.0
    joint_pos: float = 1.0
    joint_vel: float = 0.05
    actions: float = 1.0
    height_scan: float = 1.0


@configclass
class NormalizationCfg:
    obs_scales: ObsScalesCfg = ObsScalesCfg()
    clip_observations: float = 100.0
    clip_actions: float = 100.0
    height_scan_offset: float = 0.5


@configclass
class CommandRangesCfg:
    lin_vel_x=(-0.5, 0.5),
    lin_vel_y=(-0.5, 0.5),
    height=(0.72, 0.82),
    link_height=(0.4, 0.9),
    ang_vel_z=(-0.0, 0.0),
    heading: tuple = (0, 0)


@configclass
class PoseCommandRangesCfg:
    left_r = (-2.14, -0.98),
    left_p = (-0.81, 0.81),
    left_yaw = (-0.78, 0.2),
    right_r = (0.98, 2.14),
    right_p = (-0.81, 0.81),
    right_yaw = (-0.2, 0.78),
    left_x = (0.12, 0.32),
    left_y = (0.02, 0.32),
    left_z = (-0.02, 0.28),
    right_x = (0.12, 0.32),
    right_y = (-0.32, -0.02),
    right_z = (-0.02, 0.28),


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (2.0, 2.0)
    rel_standing_envs: float = 0.3
    rel_heading_envs: float = 0.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class PoseCommandsCfg:
    resampling_time_range: tuple = (0.1, 0.1)
    debug_vis: bool = True
    ranges: PoseCommandRangesCfg = PoseCommandRangesCfg()


@configclass
class NoiseScalesCfg:
    ang_vel: float = 0.2
    projected_gravity: float = 0.05
    joint_pos: float = 0.01
    joint_vel: float = 1.5
    height_scan: float = 0.1


@configclass
class NoiseCfg:
    add_noise: bool = True
    noise_scales: NoiseScalesCfg = NoiseScalesCfg()


@configclass
class CollaborationEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.4),
            "dynamic_friction_range": (0.4, 1.2),
            "restitution_range": (0.0, 0.2),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (-2.0, 2.0),
            "operation": "add",
        },
    )
    randomize_coms_without_inertia=EventTerm(
        func=mdp.randomize_coms_without_inertia,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "coms_offset_distribution_params": (-0.02, 0.02),
            "operation": "add",
        },
    )

    reset_body_link_mass=EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    reset_hand_mass=EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )

    reset_target_object_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("my_cube_sphere_art", body_names="top_box"),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    ),


    reset_root_and_box_link_setting_with_joint_state_uniform=EventTerm(
        func=mdp.reset_root_and_box_link_setting_with_joint_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
            "joint_position_range": (1.0, 1.0),
            "joint_velocity_range": (0.0, 0.0),
        },
    )


    randomize_joint_gains=EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )


    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (0.0, 0.0),
        },
    )


    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ActionDelayCfg:
    enable: bool = True
    params: dict = {"max_delay": 2, "min_delay": 0}


@configclass
class DomainRandCfg:
    events: CollaborationEventCfg = CollaborationEventCfg()
    action_delay: ActionDelayCfg = ActionDelayCfg()


@configclass
class PhysxCfg:
    gpu_max_rigid_patch_count: int = 10 * 2**15


@configclass
class SimCfg:
    dt: float = 0.0025
    decimation: int = 8
    physx: PhysxCfg = PhysxCfg()

@configclass
class CollaborationEnvCfg:
    experiment: object = MISSING
    device: str = "cuda:0"
    scene: CollaborationSceneSettingsCfg = CollaborationSceneSettingsCfg(
        max_episode_length_s=20.0,
        num_envs=4096,
        env_spacing=2.5,
        robot=MISSING,
        terrain_type=MISSING,
        terrain_generator=None,
        max_init_terrain_level=5,
        height_scanner=HeightScannerCfg(
            enable_height_scan=False,
            prim_body_name=MISSING,
            resolution=0.1,
            size=(1.6, 1.0),
            debug_vis=False,
            drift_range=(0.0, 0.0)
        )
    )
    robot: object = MISSING
    support_height_control: SupportHeightControlCfg = MISSING
    bar_controller: BarControllerCfg = MISSING
    fixed_bar_side: str = "left"
    reward = CollaborationRewardBaseCfg()
    normalization: NormalizationCfg = NormalizationCfg(
        obs_scales=ObsScalesCfg(
            lin_vel=1.0,
            ang_vel=0.25,
            projected_gravity=1.0,
            commands=1.0,
            joint_pos=1.0,
            joint_vel=0.05,
            actions=1.0,
            height_scan=1.0,
        ),
        clip_observations=100.0,
        clip_actions=100.0,
        height_scan_offset=0.5
    )
    commands: CommandsCfg = CommandsCfg(
        resampling_time_range=(8.0, 8.0),
        rel_standing_envs=0.3,
        rel_heading_envs=0.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.5, 0.5),
            lin_vel_y=(-0.5, 0.5),
            height=(0.72, 0.82),
            link_height=(0.4, 0.9),
            ang_vel_z=(-0.0, 0.0),
            heading=(0, 0)
        ),
    )
    pose_commands: PoseCommandsCfg = PoseCommandsCfg(
        resampling_time_range=(0.1, 0.1),
        debug_vis=True,
        ranges=PoseCommandRangesCfg(
            left_r = (-2.14, -0.98),
            left_p = (-0.81, 0.81),
            left_yaw = (-0.78, 0.2),
            right_r = (0.98, 2.14),
            right_p = (-0.81, 0.81),
            right_yaw = (-0.2, 0.78),
            left_x = (0.12, 0.32),
            left_y = (0.02, 0.32),
            left_z = (-0.02, 0.28),
            right_x = (0.12, 0.32),
            right_y = (-0.32, -0.02),
            right_z = (-0.02, 0.28),
        ),
    )
    noise: NoiseCfg = NoiseCfg(
        add_noise=True,
        noise_scales=NoiseScalesCfg(
            ang_vel=0.2,
            projected_gravity=0.05,
            joint_pos=0.01,
            joint_vel=1.5,
            height_scan=0.1,
        )
    )
    domain_rand: DomainRandCfg = DomainRandCfg(
        events=CollaborationEventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.4),
                    "dynamic_friction_range": (0.4, 1.2),
                    "restitution_range": (0.0, 0.02),
                    "num_buckets": 64,
                },
            ),
            add_base_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                    "mass_distribution_params": (-2.0, 2.0),
                    "operation": "add",
                },
            ),
            randomize_coms_without_inertia=EventTerm(
                func=mdp.randomize_coms_without_inertia,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                    "coms_offset_distribution_params": (-0.02, 0.02),
                    "operation": "add",
                },
            ),
            reset_body_link_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                    "mass_distribution_params": (0.9, 1.1),
                    "operation": "scale",
                },
            ),
            reset_hand_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                    "mass_distribution_params": (0.9, 1.1),
                    "operation": "scale",
                },
            ),
            reset_target_object_mass=EventTerm(
                func=mdp.randomize_rigid_body_mass,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("my_cube_sphere_art", body_names="top_box"),
                    "mass_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ),
            reset_root_and_box_link_setting_with_joint_state_uniform=EventTerm(
                func=mdp.reset_root_and_box_link_setting_with_joint_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
                    "velocity_range": {
                        "x": (-0.5, 0.5),
                        "y": (-0.5, 0.5),
                        "z": (-0.5, 0.5),
                        "roll": (-0.5, 0.5),
                        "pitch": (-0.5, 0.5),
                        "yaw": (-0.5, 0.5),
                    },
                    "joint_position_range": (1.0, 1.0),
                    "joint_velocity_range": (0.0, 0.0),
                },
            ),


            randomize_joint_gains=EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
                    "stiffness_distribution_params": (0.8, 1.2),
                    "damping_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ),




            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (1.0, 1.0),
                    "velocity_range": (0.0, 0.0),
                },
            ),


            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 15.0),
                params={
                    "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ),
        ),
        action_delay=ActionDelayCfg(
            enable=True,
            params={"max_delay": 1, "min_delay": 0}
        ),
    )
    sim: SimCfg = SimCfg(
        dt=0.0025,
        decimation=8,
        physx=PhysxCfg(
            gpu_max_rigid_patch_count=10 * 2**15
        )
    )

    def __post_init__(self):
        pass


@configclass
class TeacherAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 20000
    empirical_normalization = False
    policy = RslRlPpoActorCriticResidualCfg(
        class_name="ActorCriticWbcEnd2endFollowingWholePipeQuatResiVel29",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        history_length=10,
        num_envs=MISSING,
        residual_hidden_init_std=MISSING,
        residual_final_init_std=MISSING,
        residual_bias_init=MISSING,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO_WbcEnd2endWholePipeResiVel",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        normalize_advantage_per_mini_batch=False,
        adaptive_kl_high_factor=MISSING,
        adaptive_kl_low_factor=MISSING,
        adaptive_lr_factor=MISSING,
        min_learning_rate=MISSING,
        max_learning_rate=MISSING,
        symmetry_cfg=None,
        rnd_cfg=None,
    )
    clip_actions = None
    save_interval = 1500
    experiment_name = ""
    run_name = ""
    logger = "wandb"
    neptune_project = "leggedlab"
    wandb_project = "leggedlab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    def __post_init__(self):
        pass

@configclass
class StudentAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 10000
    empirical_normalization = False
    policy = RslRlDistillationStudentTeacherCfg(
        class_name="StudentTeacherDistill",
        init_noise_std=0.1,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        teacher_action_clip=MISSING,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        class_name="DistillationDistill",
        num_learning_epochs=2,
        gradient_length=15,
        learning_rate=1e-3,
    )
    clip_actions = None
    save_interval = 200
    experiment_name = ""
    run_name = ""
    logger = "wandb"
    neptune_project = "leggedlab"
    wandb_project = "leggedlab"
    resume = False
    load_run = ".*"
    load_checkpoint = "model_.*.pt"

    def __post_init__(self):
        pass


def collaboration_experiment_cfg(
    *, masked_height_command: float | None, vertical_velocity_noise: float
) -> CollaborationExperimentCfg:
    """Build the observation, grasp, command, and reset contract shared by COLA."""

    return CollaborationExperimentCfg(
        termination_observation=TerminationObservationCfg(
            robot_contact_force_threshold=1.0,
            feet_contact_force_threshold=0.5,
            object_min_height=0.25,
        ),
        grasp=CollaborationGraspCfg(
            left_palm_quat=(
                0.70712793,
                -0.70708555,
                0.00008733,
                -0.00004816,
            ),
            right_palm_quat=(
                0.70712793,
                0.70708561,
                0.00008731,
                0.00004810,
            ),
            left_palm_xyz=(0.2413, 0.1517, 0.0952),
            right_palm_xyz=(0.2413, -0.1516, 0.0952),
        ),
        observations=CollaborationObservationCfg(
            mask_planar_velocity_command=True,
            masked_height_command=masked_height_command,
        ),
        support_motion=SupportMotionCfg(
            no_object_probability=0.0,
            bar_height_command_offset=0.06,
            bar_height_command_range=(0.55, 0.90),
            no_object_height_command=0.78,
            planar_velocity_noise=0.1,
            vertical_velocity_noise=vertical_velocity_noise,
            angular_velocity_noise=0.1,
            angular_velocity_noise_interval_steps=10,
            angular_velocity_noise_activation_probability=0.5,
        ),
        support_reset=SupportResetCfg(
            positive_y_probability=0.0,
            root_x_jitter_range=(0.0, 0.0),
            root_y_jitter_range=(0.0, 0.0),
        ),
    )


def support_height_control_cfg() -> SupportHeightControlCfg:
    return SupportHeightControlCfg(
        ankle_height_offset=0.085,
        ankle_height_correction_threshold=0.1,
        corrected_height_range=(0.25, 1.0),
        vertical_joint_height_offset=0.82,
        vertical_joint_velocity_gain=0.25,
    )


def bar_controller_cfg() -> BarControllerCfg:
    return BarControllerCfg(
        yaw_kp=40.0,
        yaw_kd=4.0,
        yaw_torque_limit=10.0,
        gain_randomization_enabled=True,
    )


def configure_fixed_bar_g1(cfg, *, robot_asset, terrain, debug_vis: bool) -> None:
    """Apply the G1 asset, terrain, command, and randomization settings."""

    cfg.commands.debug_vis = debug_vis
    cfg.pose_commands.debug_vis = debug_vis
    cfg.scene.height_scanner.prim_body_name = "torso_link"
    cfg.scene.robot = robot_asset
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    cfg.scene.terrain_type = "generator"
    cfg.scene.terrain_generator = terrain
    nominal_height = cfg.experiment.support_motion.no_object_height_command
    cfg.commands.resampling_time_range = (4.0, 12.0)
    cfg.commands.ranges.height = (nominal_height, nominal_height)
    cfg.commands.ranges.link_height = (0.50, 0.90)
    if hasattr(cfg.robot, "student_obs_history_length"):
        cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.robot.terminate_contacts_body_names = ["^(?!.*(ankle|hand)).*$"]
    cfg.robot.feet_body_names = [".*ankle_roll.*"]

    events = cfg.domain_rand.events
    events.add_base_mass.params["asset_cfg"].body_names = [".*torso_link.*"]
    events.randomize_joint_gains.params["asset_cfg"] = SceneEntityCfg("robot")
    events.randomize_coms_without_inertia.params["asset_cfg"].body_names = [
        ".*torso_link.*"
    ]
    events.reset_body_link_mass.params["asset_cfg"].body_names = [
        "left_hip_yaw_link",
        "left_hip_roll_link",
        "left_hip_pitch_link",
        "right_hip_yaw_link",
        "right_hip_roll_link",
        "right_hip_pitch_link",
    ]
    events.reset_hand_mass = None
    mass_event = events.reset_target_object_mass
    mass_event.params["asset_cfg"] = SceneEntityCfg("carried_bar")
    mass_event.params["mass_distribution_params"] = (0.8, 1.2)
    mass_event.params["operation"] = "scale"
    events.reset_root_and_box_link_setting_with_joint_state_uniform.func = (
        collaboration_mdp.reset_robot_and_bar_uniform
    )


def _robot_root_reset_event() -> EventTerm:
    return EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )


def _hand_mass_event() -> EventTerm:
    return EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=[".*hand_palm_link"]
            ),
            "mass_distribution_params": (0.9, 1.1),
            "operation": "scale",
        },
    )


def configure_no_object_g1(cfg, *, robot_asset, terrain, debug_vis: bool) -> None:
    configure_fixed_bar_g1(
        cfg,
        robot_asset=robot_asset,
        terrain=terrain,
        debug_vis=debug_vis,
    )
    nominal_height = cfg.experiment.support_motion.no_object_height_command
    cfg.commands.rel_standing_envs = 1.0
    cfg.commands.rel_heading_envs = 0.0
    cfg.commands.heading_command = False
    cfg.commands.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.ranges.height = (nominal_height, nominal_height)
    cfg.commands.ranges.link_height = (nominal_height, nominal_height)
    cfg.commands.ranges.heading = None

    events = cfg.domain_rand.events
    events.reset_target_object_mass = None
    events.reset_root_and_box_link_setting_with_joint_state_uniform = (
        _robot_root_reset_event()
    )
    events.reset_hand_mass = _hand_mass_event()


def configure_teacher_agent(cfg) -> None:
    cfg.policy.residual_hidden_init_std = 0.01
    cfg.policy.residual_final_init_std = 0.0
    cfg.policy.residual_bias_init = 0.0
    cfg.policy.base_privileged_obs_per_frame = 20
    cfg.algorithm.adaptive_kl_high_factor = 2.0
    cfg.algorithm.adaptive_kl_low_factor = 2.0
    cfg.algorithm.adaptive_lr_factor = 1.5
    cfg.algorithm.min_learning_rate = 1.0e-5
    cfg.algorithm.max_learning_rate = 1.0e-2


def configure_student_agent(cfg) -> None:
    cfg.policy.teacher_action_clip = 15.0
    cfg.policy.teacher_base_privileged_obs_per_frame = 20


__all__ = [
    "bar_controller_cfg",
    "CollaborationEnvCfg",
    "CollaborationRewardBaseCfg",
    "collaboration_experiment_cfg",
    "configure_fixed_bar_g1",
    "configure_no_object_g1",
    "configure_student_agent",
    "configure_teacher_agent",
    "student_robot_cfg",
    "StudentAgentCfg",
    "support_height_control_cfg",
    "teacher_robot_cfg",
    "TeacherAgentCfg",
]
