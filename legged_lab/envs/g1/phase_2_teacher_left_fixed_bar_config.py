"""Phase-2 teacher with the bar fixed to the left wrist."""

from isaaclab.utils import configclass

from legged_lab.assets.unitree import G1_DIRECT_FIXED_BAR_CFG
from legged_lab.envs.base.collaboration_config import (
    CollaborationEnvCfg,
    TeacherAgentCfg,
    bar_controller_cfg,
    collaboration_experiment_cfg,
    configure_teacher_agent,
    configure_fixed_bar_g1,
    support_height_control_cfg,
    teacher_robot_cfg,
)
from legged_lab.envs.base.reward_config import TeacherFixedBarRewardCfg
from legged_lab.terrains import ROUGH_EASY_TERRAINS_NO_STAIRS_EASY_CFG


@configclass
class Phase2TeacherLeftFixedBarEnvCfg(CollaborationEnvCfg):
    robot = teacher_robot_cfg()
    reward = TeacherFixedBarRewardCfg()
    support_height_control = support_height_control_cfg()
    experiment = collaboration_experiment_cfg(
        masked_height_command=None,
        vertical_velocity_noise=0.0,
    )
    bar_controller = bar_controller_cfg()
    fixed_bar_side: str = "left"

    def __post_init__(self):
        super().__post_init__()
        configure_fixed_bar_g1(
            self,
            robot_asset=G1_DIRECT_FIXED_BAR_CFG,
            terrain=ROUGH_EASY_TERRAINS_NO_STAIRS_EASY_CFG,
            debug_vis=True,
        )


@configclass
class Phase2TeacherLeftFixedBarAgentCfg(TeacherAgentCfg):
    experiment_name: str = "cola_phase_2_teacher_left_fixed_bar"
    wandb_project: str = "cola_phase_2_teacher_left_fixed_bar"

    def __post_init__(self):
        super().__post_init__()
        configure_teacher_agent(self)


__all__ = ["Phase2TeacherLeftFixedBarAgentCfg", "Phase2TeacherLeftFixedBarEnvCfg"]
