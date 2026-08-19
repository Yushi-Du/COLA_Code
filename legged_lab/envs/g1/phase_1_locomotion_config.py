"""Phase-1 whole-body locomotion configuration."""

from isaaclab.utils import configclass

from legged_lab.assets.unitree import G1_HAND_FIXED_CFG
from legged_lab.envs.base.common_config import (
    LocomotionExperimentCfg,
    LocomotionPoseTrajectoryCfg,
    TerminationObservationCfg,
)
from legged_lab.envs.base.locomotion_config import (
    LocomotionAgentCfg,
    LocomotionEnvCfg,
)
from legged_lab.envs.base.reward_config import Phase1RewardCfg
from legged_lab.terrains import PHASE1_TERRAINS_CFG


@configclass
class Phase1LocomotionEnvCfg(LocomotionEnvCfg):
    reward = Phase1RewardCfg()
    experiment = LocomotionExperimentCfg(
        termination_observation=TerminationObservationCfg(
            robot_contact_force_threshold=1.0,
            feet_contact_force_threshold=0.5,
            object_min_height=None,
        ),
        pose_trajectory=LocomotionPoseTrajectoryCfg(
            num_waypoints=10,
            lateral_randomization_scale=1.0,
            vertical_randomization_scale=1.0,
            left_center_xyz=(0.2413, 0.1517, 0.0952),
            right_center_xyz=(0.2413, -0.1516, 0.0952),
            left_center_quat=(0.707, -0.707, 0.0, 0.0),
            right_center_quat=(0.707, 0.707, 0.0, 0.0),
        ),
    )

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner.prim_body_name = "torso_link"
        self.scene.robot = G1_HAND_FIXED_CFG
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = True
        self.scene.terrain_type = "generator"
        self.scene.terrain_generator = PHASE1_TERRAINS_CFG
        self.robot.terminate_contacts_body_names = ["^(?!.*(ankle|hand)).*$"]
        self.robot.feet_body_names = [".*ankle_roll.*"]

        events = self.domain_rand.events
        events.add_base_mass.params["asset_cfg"].body_names = [".*torso_link.*"]
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
        events.reset_hand_mass.params["asset_cfg"].body_names = [
            ".*hand_palm_link"
        ]


@configclass
class Phase1LocomotionAgentCfg(LocomotionAgentCfg):
    experiment_name: str = "cola_phase_1_locomotion"
    wandb_project: str = "cola_phase_1_locomotion"

    def __post_init__(self):
        super().__post_init__()
        self.algorithm.adaptive_kl_high_factor = 2.0
        self.algorithm.adaptive_kl_low_factor = 2.0
        self.algorithm.adaptive_lr_factor = 1.5
        self.algorithm.min_learning_rate = 1.0e-5
        self.algorithm.max_learning_rate = 1.0e-2


__all__ = ["Phase1LocomotionAgentCfg", "Phase1LocomotionEnvCfg"]
