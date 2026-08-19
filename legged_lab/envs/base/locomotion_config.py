"""Configuration schemas for phase-1 whole-body locomotion."""

from dataclasses import MISSING
import math

import numpy as np
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass

import legged_lab.mdp as mdp
from legged_lab.utils.rsl_rl_cfg import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticWbcEnd2endFollowingCfg,
    RslRlPpoAlgorithmCfg,
)

@configclass
class LocomotionRewardCfg:
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
class LocomotionSceneSettingsCfg:
    max_episode_length_s: float = 20.0
    num_envs: int = 4096
    env_spacing: float = 2.5
    robot: ArticulationCfg = MISSING
    terrain_type: str = MISSING
    terrain_generator: TerrainGeneratorCfg = None
    max_init_terrain_level: int = 5
    height_scanner: HeightScannerCfg = HeightScannerCfg()


@configclass
class LocomotionRobotCfg:
    actor_obs_history_length: int = 10
    critic_obs_history_length: int = 10
    action_scale: float = 0.25
    terminate_contacts_body_names: list = []
    feet_body_names: list = []


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
    lin_vel_x=(-0.75, 1.05),
    lin_vel_y=(-0.5, 0.5),
    height=(0.45, 0.9),
    ang_vel_z=(-1.2, 1.2),
    heading: tuple = (0, 0)


@configclass
class PoseCommandRangesCfg:
    orientation_cone_rad=np.pi / 7.0,
    cube_size=0.1


@configclass
class CommandsCfg:
    resampling_time_range: tuple = (4.0, 4.0)
    rel_standing_envs: float = 0.3
    rel_heading_envs: float = 0.0
    heading_command: bool = True
    heading_control_stiffness: float = 0.5
    debug_vis: bool = True
    ranges: CommandRangesCfg = CommandRangesCfg()


@configclass
class PoseCommandsCfg:
    resampling_time_range: tuple = (1.0, 1.0)
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
class LocomotionEventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 1.0),
            "restitution_range": (0.0, 0.02),
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
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
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
        },
    )

    randomize_joint_gains=EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )


    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 12.0),
        params={
            "velocity_range": {"x": (-1.0, 1.0), "y": (-1.0, 1.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class ActionDelayCfg:
    enable: bool = True
    params: dict = {"max_delay": 1, "min_delay": 0}


@configclass
class DomainRandCfg:
    events: LocomotionEventCfg = LocomotionEventCfg()
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
class LocomotionEnvCfg:
    experiment: object = MISSING
    device: str = "cuda:0"
    scene: LocomotionSceneSettingsCfg = LocomotionSceneSettingsCfg(
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
    robot: LocomotionRobotCfg = LocomotionRobotCfg(
        actor_obs_history_length=10,
        critic_obs_history_length=10,
        action_scale=0.25,
        terminate_contacts_body_names=MISSING,
        feet_body_names=MISSING,
    )
    reward = LocomotionRewardCfg()
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
        resampling_time_range=(4.0, 4.0),
        rel_standing_envs=0.3,
        rel_heading_envs=0.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=False,
        ranges=CommandRangesCfg(
            lin_vel_x=(-0.75, 1.05),
            lin_vel_y=(-0.5, 0.5),
            height=(0.45, 0.9),
            ang_vel_z=(-1.2, 1.2),
            heading=(0, 0)
        ),
    )
    pose_commands: PoseCommandsCfg = PoseCommandsCfg(
        resampling_time_range=(1.0, 1.0),
        debug_vis=False,
        ranges=PoseCommandRangesCfg(
            orientation_cone_rad=np.pi / 7.0,
            cube_size=0.1
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
        events=LocomotionEventCfg(
            physics_material=EventTerm(
                func=mdp.randomize_rigid_body_material,
                mode="startup",
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                    "static_friction_range": (0.6, 1.0),
                    "dynamic_friction_range": (0.4, 1.0),
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
            reset_base=EventTerm(
                func=mdp.reset_root_state_uniform,
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
                },
            ),

            randomize_joint_gains=EventTerm(
                func=mdp.randomize_actuator_gains,
                mode="reset",
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "stiffness_distribution_params": (0.8, 1.2),
                    "damping_distribution_params": (0.8, 1.2),
                    "operation": "scale",
                },
            ),

            reset_robot_joints=EventTerm(
                func=mdp.reset_joints_by_scale,
                mode="reset",
                params={
                    "position_range": (0.5, 1.5),
                    "velocity_range": (0.0, 0.0),
                },
            ),


            push_robot=EventTerm(
                func=mdp.push_by_setting_velocity,
                mode="interval",
                interval_range_s=(10.0, 12.0),
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
class LocomotionAgentCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    device = "cuda:0"
    num_steps_per_env = 24
    max_iterations = 30000
    empirical_normalization = False
    policy = RslRlPpoActorCriticWbcEnd2endFollowingCfg(
        class_name="ActorCriticWbcEnd2endQuat",
        init_noise_std=1.0,
        noise_std_type="scalar",
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        history_length=10,
        num_envs=MISSING,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="PPO_WbcEnd2endQuat",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
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
    save_interval = 2000
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


__all__ = [
    "LocomotionAgentCfg",
    "LocomotionEnvCfg",
    "LocomotionRewardCfg",
]
