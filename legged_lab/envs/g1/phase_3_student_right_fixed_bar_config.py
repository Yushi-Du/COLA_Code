"""Phase-3 student with the bar fixed to the right wrist."""

from isaaclab.utils import configclass

from legged_lab.assets.unitree import G1_DIRECT_FIXED_BAR_CFG
from legged_lab.envs.base.collaboration_config import (
    CollaborationEnvCfg,
    StudentAgentCfg,
    bar_controller_cfg,
    collaboration_experiment_cfg,
    configure_student_agent,
    configure_fixed_bar_g1,
    student_robot_cfg,
    support_height_control_cfg,
)
from legged_lab.envs.base.reward_config import StudentFixedBarRewardCfg
from legged_lab.terrains import FLAT_TERRAINS_CFG


@configclass
class Phase3StudentRightFixedBarEnvCfg(CollaborationEnvCfg):
    robot = student_robot_cfg()
    reward = StudentFixedBarRewardCfg()
    support_height_control = support_height_control_cfg()
    experiment = collaboration_experiment_cfg(
        masked_height_command=0.78,
        vertical_velocity_noise=0.1,
        student_mass_observation_enabled=True,
    )
    bar_controller = bar_controller_cfg()
    fixed_bar_side: str = "right"

    def __post_init__(self):
        super().__post_init__()
        configure_fixed_bar_g1(
            self,
            robot_asset=G1_DIRECT_FIXED_BAR_CFG,
            terrain=FLAT_TERRAINS_CFG,
            debug_vis=False,
        )


@configclass
class Phase3StudentRightFixedBarAgentCfg(StudentAgentCfg):
    experiment_name: str = "cola_phase_3_student_right_fixed_bar"
    wandb_project: str = "cola_phase_3_student_right_fixed_bar"

    def __post_init__(self):
        super().__post_init__()
        configure_student_agent(self)


__all__ = ["Phase3StudentRightFixedBarAgentCfg", "Phase3StudentRightFixedBarEnvCfg"]
