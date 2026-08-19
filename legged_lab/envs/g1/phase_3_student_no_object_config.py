"""Phase-3 student fixed-hand population without a carried object."""

from isaaclab.utils import configclass

from legged_lab.assets.unitree import G1_HAND_FIXED_CFG
from legged_lab.envs.base.collaboration_config import (
    CollaborationEnvCfg,
    StudentAgentCfg,
    bar_controller_cfg,
    collaboration_experiment_cfg,
    configure_student_agent,
    configure_no_object_g1,
    student_robot_cfg,
    support_height_control_cfg,
)
from legged_lab.envs.base.reward_config import StudentNoObjectRewardCfg
from legged_lab.terrains import FLAT_TERRAINS_CFG


@configclass
class Phase3StudentNoObjectEnvCfg(CollaborationEnvCfg):
    robot = student_robot_cfg()
    reward = StudentNoObjectRewardCfg()
    support_height_control = support_height_control_cfg()
    experiment = collaboration_experiment_cfg(
        masked_height_command=0.78,
        vertical_velocity_noise=0.1,
    )
    bar_controller = bar_controller_cfg()

    def __post_init__(self):
        super().__post_init__()
        configure_no_object_g1(
            self,
            robot_asset=G1_HAND_FIXED_CFG,
            terrain=FLAT_TERRAINS_CFG,
            debug_vis=False,
        )


@configclass
class Phase3StudentNoObjectAgentCfg(StudentAgentCfg):
    experiment_name: str = "cola_phase_3_student_no_object"
    wandb_project: str = "cola_phase_3_student_no_object"

    def __post_init__(self):
        super().__post_init__()
        configure_student_agent(self)


__all__ = ["Phase3StudentNoObjectAgentCfg", "Phase3StudentNoObjectEnvCfg"]
