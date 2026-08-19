"""Registry for the seven released COLA tasks."""

from legged_lab.envs.base.collaboration_env import FixedBarCollaborationEnv
from legged_lab.envs.base.locomotion_env import LocomotionEnv
from legged_lab.envs.base.no_object_env import NoObjectEnv
from legged_lab.envs.g1.phase_1_locomotion_config import (
    Phase1LocomotionAgentCfg,
    Phase1LocomotionEnvCfg,
)
from legged_lab.envs.g1.phase_2_teacher_left_fixed_bar_config import (
    Phase2TeacherLeftFixedBarAgentCfg,
    Phase2TeacherLeftFixedBarEnvCfg,
)
from legged_lab.envs.g1.phase_2_teacher_no_object_config import (
    Phase2TeacherNoObjectAgentCfg,
    Phase2TeacherNoObjectEnvCfg,
)
from legged_lab.envs.g1.phase_2_teacher_right_fixed_bar_config import (
    Phase2TeacherRightFixedBarAgentCfg,
    Phase2TeacherRightFixedBarEnvCfg,
)
from legged_lab.envs.g1.phase_3_student_left_fixed_bar_config import (
    Phase3StudentLeftFixedBarAgentCfg,
    Phase3StudentLeftFixedBarEnvCfg,
)
from legged_lab.envs.g1.phase_3_student_no_object_config import (
    Phase3StudentNoObjectAgentCfg,
    Phase3StudentNoObjectEnvCfg,
)
from legged_lab.envs.g1.phase_3_student_right_fixed_bar_config import (
    Phase3StudentRightFixedBarAgentCfg,
    Phase3StudentRightFixedBarEnvCfg,
)
from legged_lab.utils.task_registry import task_registry


task_registry.register(
    "cola_phase_1_locomotion",
    LocomotionEnv,
    Phase1LocomotionEnvCfg(),
    Phase1LocomotionAgentCfg(),
)
task_registry.register(
    "cola_phase_2_teacher_left_fixed_bar",
    FixedBarCollaborationEnv,
    Phase2TeacherLeftFixedBarEnvCfg(),
    Phase2TeacherLeftFixedBarAgentCfg(),
)
task_registry.register(
    "cola_phase_2_teacher_right_fixed_bar",
    FixedBarCollaborationEnv,
    Phase2TeacherRightFixedBarEnvCfg(),
    Phase2TeacherRightFixedBarAgentCfg(),
)
task_registry.register(
    "cola_phase_2_teacher_no_object",
    NoObjectEnv,
    Phase2TeacherNoObjectEnvCfg(),
    Phase2TeacherNoObjectAgentCfg(),
)
task_registry.register(
    "cola_phase_3_student_left_fixed_bar",
    FixedBarCollaborationEnv,
    Phase3StudentLeftFixedBarEnvCfg(),
    Phase3StudentLeftFixedBarAgentCfg(),
)
task_registry.register(
    "cola_phase_3_student_right_fixed_bar",
    FixedBarCollaborationEnv,
    Phase3StudentRightFixedBarEnvCfg(),
    Phase3StudentRightFixedBarAgentCfg(),
)
task_registry.register(
    "cola_phase_3_student_no_object",
    NoObjectEnv,
    Phase3StudentNoObjectEnvCfg(),
    Phase3StudentNoObjectAgentCfg(),
)


__all__ = []
