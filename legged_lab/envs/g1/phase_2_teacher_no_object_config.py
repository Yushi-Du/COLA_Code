"""Phase-2 teacher fixed-hand population without a carried object."""

from isaaclab.utils import configclass

from legged_lab.assets.unitree import G1_HAND_FIXED_CFG
from legged_lab.envs.base.collaboration_config import (
    CollaborationEnvCfg,
    TeacherAgentCfg,
    bar_controller_cfg,
    collaboration_experiment_cfg,
    configure_teacher_agent,
    configure_no_object_g1,
    support_height_control_cfg,
    teacher_robot_cfg,
)
from legged_lab.envs.base.reward_config import TeacherNoObjectRewardCfg
from legged_lab.terrains import ROUGH_EASY_TERRAINS_NO_STAIRS_EASY_CFG


@configclass
class Phase2TeacherNoObjectEnvCfg(CollaborationEnvCfg):
    robot = teacher_robot_cfg()
    reward = TeacherNoObjectRewardCfg()
    support_height_control = support_height_control_cfg()
    experiment = collaboration_experiment_cfg(
        masked_height_command=None,
        vertical_velocity_noise=0.0,
    )
    bar_controller = bar_controller_cfg()

    def __post_init__(self):
        super().__post_init__()
        configure_no_object_g1(
            self,
            robot_asset=G1_HAND_FIXED_CFG,
            terrain=ROUGH_EASY_TERRAINS_NO_STAIRS_EASY_CFG,
            debug_vis=False,
        )


@configclass
class Phase2TeacherNoObjectAgentCfg(TeacherAgentCfg):
    experiment_name: str = "cola_phase_2_teacher_no_object"
    wandb_project: str = "cola_phase_2_teacher_no_object"

    def __post_init__(self):
        super().__post_init__()
        configure_teacher_agent(self)


__all__ = ["Phase2TeacherNoObjectAgentCfg", "Phase2TeacherNoObjectEnvCfg"]
