"""COLA-specific domain-randomization and paired-reset events."""

from __future__ import annotations

from typing import Literal

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils


def randomize_coms_without_inertia(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    coms_offset_distribution_params: tuple[float, float],
    operation: Literal["add"] = "add",
    distribution: Literal["uniform"] = "uniform",
) -> None:
    """Randomize body CoMs from their startup values without changing inertia.

    The active COLA configs use uniform additive offsets. Keeping the supported
    surface explicit avoids cumulative offsets across repeated startup calls.
    """

    if operation != "add" or distribution != "uniform":
        raise ValueError("COLA CoM randomization supports only uniform additive offsets")

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids_cpu = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.long, device="cpu")
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, dtype=torch.long, device="cpu")

    if env.default_coms is None:
        env.default_coms = asset.root_physx_view.get_coms().clone()
    coms = env.default_coms.clone()
    low, high = coms_offset_distribution_params
    offsets = torch.empty(
        (len(env_ids_cpu), len(body_ids), 3), dtype=coms.dtype, device="cpu"
    ).uniform_(low, high)
    # Isaac Sim 5.1 exposes each CoM as a 7-D pose (xyz + quaternion).
    # Only the translational component is randomized.
    coms[env_ids_cpu[:, None], body_ids, :3] += offsets
    asset.root_physx_view.set_coms(coms, env_ids_cpu)


def reset_root_and_box_link_setting_with_joint_state_uniform(
    env,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    joint_position_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset the robot and driven partner/support while preserving their pose."""

    asset: RigidObject | Articulation = env.scene[asset_cfg.name]
    root_states = asset.data.default_root_state[env_ids].clone()
    support_states = env.my_cube_sphere_art.data.default_root_state[env_ids].clone()

    reset_cfg = env.cfg.experiment.support_reset
    signs = torch.where(
        torch.rand(len(env_ids), device=support_states.device)
        < reset_cfg.positive_y_probability,
        1.0,
        -1.0,
    )
    support_states[:, 1] = torch.abs(support_states[:, 1]) * signs
    support_states[:, 0] += math_utils.sample_uniform(
        *reset_cfg.root_x_jitter_range,
        (len(env_ids),),
        support_states.device,
    )
    support_states[:, 1] += math_utils.sample_uniform(
        *reset_cfg.root_y_jitter_range,
        (len(env_ids),),
        support_states.device,
    )

    joint_pos = env.my_cube_sphere_art.data.default_joint_pos[env_ids].clone()
    joint_vel = env.my_cube_sphere_art.data.default_joint_vel[env_ids].clone()
    # ``signs`` is the desired Y side of the support root.  The source USD's
    # default root is at -Y and its default -pi/2 top joint points the bar
    # toward the robot.  Therefore the joint mirror has the opposite sign to
    # the requested root side.  Using the same sign points the bar outward.
    joint_pos *= -signs.unsqueeze(-1)
    joint_pos *= math_utils.sample_uniform(
        *joint_position_range, joint_pos.shape, joint_pos.device
    )
    joint_vel *= math_utils.sample_uniform(
        *joint_velocity_range, joint_vel.shape, joint_vel.device
    )
    joint_pos_limits = env.my_cube_sphere_art.data.soft_joint_pos_limits[env_ids]
    joint_pos.clamp_(joint_pos_limits[..., 0], joint_pos_limits[..., 1])
    joint_vel_limits = env.my_cube_sphere_art.data.soft_joint_vel_limits[env_ids]
    joint_vel.clamp_(-joint_vel_limits, joint_vel_limits)
    env.my_cube_sphere_art.write_joint_state_to_sim(
        joint_pos, joint_vel, env_ids=env_ids
    )

    relative_position = support_states[:, :3] - root_states[:, :3]
    relative_quat = math_utils.quat_mul(
        math_utils.quat_conjugate(root_states[:, 3:7]), support_states[:, 3:7]
    )

    pose_ranges = torch.tensor(
        [pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=asset.device,
    )
    pose_samples = math_utils.sample_uniform(
        pose_ranges[:, 0], pose_ranges[:, 1], (len(env_ids), 6), device=asset.device
    )
    positions = root_states[:, :3] + env.scene.env_origins[env_ids] + pose_samples[:, :3]
    orientation_delta = math_utils.quat_from_euler_xyz(
        pose_samples[:, 3], pose_samples[:, 4], pose_samples[:, 5]
    )
    orientations = math_utils.quat_mul(root_states[:, 3:7], orientation_delta)

    velocity_ranges = torch.tensor(
        [velocity_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
        device=asset.device,
    )
    velocity_samples = math_utils.sample_uniform(
        velocity_ranges[:, 0],
        velocity_ranges[:, 1],
        (len(env_ids), 6),
        device=asset.device,
    )
    velocities = root_states[:, 7:13] + velocity_samples
    support_velocities = support_states[:, 7:13] + velocity_samples
    support_positions = positions + math_utils.quat_rotate(orientations, relative_position)
    support_orientations = math_utils.quat_mul(orientations, relative_quat)

    asset.write_root_pose_to_sim(
        torch.cat((positions, orientations), dim=-1), env_ids=env_ids
    )
    asset.write_root_velocity_to_sim(velocities, env_ids=env_ids)
    env.my_cube_sphere_art.write_root_pose_to_sim(
        torch.cat((support_positions, support_orientations), dim=-1), env_ids=env_ids
    )
    env.my_cube_sphere_art.write_root_velocity_to_sim(
        support_velocities, env_ids=env_ids
    )
