# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Project-local RSL-RL configs required by COLA's custom runners.

The upstream 2.3 configs add normalization and observation-group fields for
modern RSL-RL. COLA uses its own runner fork and policy constructors, so those
extra keys would be forwarded to incompatible constructors. These compact
configs preserve the checkpoint-compatible dictionary interface.
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import Any, Literal

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlPpoAlgorithmCfg as IsaacLabRslRlPpoAlgorithmCfg,
)


@configclass
class RslRlPpoActorCriticWbcEnd2endFollowingCfg:
    class_name: str = "ActorCriticWbcEnd2endFollowing"
    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"
    actor_hidden_dims: list[int] = MISSING
    critic_hidden_dims: list[int] = MISSING
    activation: str = MISSING
    history_length: int = MISSING
    num_envs: int = MISSING


@configclass
class RslRlPpoActorCriticResidualCfg(RslRlPpoActorCriticWbcEnd2endFollowingCfg):
    """Initialization controls for the collaboration teacher's residual nets."""

    residual_actor_hidden_dims: list[int] | None = None
    residual_critic_hidden_dims: list[int] | None = None
    residual_hidden_init_std: float = MISSING
    residual_final_init_std: float = MISSING
    residual_bias_init: float = MISSING
    base_privileged_obs_per_frame: int = 13


@configclass
class RslRlDistillationStudentTeacherCfg:
    class_name: str = "StudentTeacher"
    init_noise_std: float = MISSING
    noise_std_type: Literal["scalar", "log"] = "scalar"
    student_hidden_dims: list[int] = MISSING
    teacher_hidden_dims: list[int] = MISSING
    teacher_residual_hidden_dims: list[int] | None = None
    activation: str = MISSING
    teacher_action_clip: float = MISSING
    teacher_base_privileged_obs_per_frame: int = 13


@configclass
class RslRlDistillationAlgorithmCfg:
    class_name: str = "Distillation"
    num_learning_epochs: int = MISSING
    learning_rate: float = MISSING
    gradient_length: int = MISSING


@configclass
class RslRlPpoAlgorithmCfg(IsaacLabRslRlPpoAlgorithmCfg):
    """PPO schema extended with COLA's config-driven adaptive-KL controls."""

    adaptive_kl_high_factor: float = MISSING
    adaptive_kl_low_factor: float = MISSING
    adaptive_lr_factor: float = MISSING
    min_learning_rate: float = MISSING
    max_learning_rate: float = MISSING


@configclass
class RslRlOnPolicyRunnerCfg:
    seed: int = 42
    device: str = "cuda:0"
    num_steps_per_env: int = MISSING
    max_iterations: int = MISSING
    empirical_normalization: bool = MISSING
    policy: Any = MISSING
    algorithm: Any = MISSING
    clip_actions: float | None = None
    save_interval: int = MISSING
    experiment_name: str = MISSING
    run_name: str = ""
    logger: Literal["tensorboard", "neptune", "wandb"] = "tensorboard"
    neptune_project: str = "isaaclab"
    wandb_project: str = "isaaclab"
    resume: bool = False
    load_run: str = ".*"
    load_checkpoint: str = "model_.*.pt"
