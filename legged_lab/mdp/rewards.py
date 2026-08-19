from __future__ import annotations


import torch


import torch.nn.functional as F


from typing import Any


from isaaclab.managers import SceneEntityCfg


from isaaclab.sensors import ContactSensor


import isaaclab.utils.math as math_utils


from isaaclab.assets import Articulation, RigidObject


from legged_lab.utils.foot_force_balance import normalized_two_foot_load_imbalance


BaseEnv = Any


def track_lin_vel_xy_yaw_frame_exp(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    vel_yaw = math_utils.quat_rotate_inverse(math_utils.yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_lin_vel_xy_yaw_frame_exp_world(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    lin_vel_error = torch.sum(torch.square(env.command_generator.command[:, :2] - asset.data.root_lin_vel_w[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def track_ang_vel_z_world_exp_world(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_generator.command[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def lin_vel_z_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def energy(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.norm(torch.abs(asset.data.applied_torque * asset.data.joint_vel), dim=-1)
    return reward


def joint_acc_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_acc[:, asset_cfg.joint_ids]), dim=1)


def action_rate_l2(env: BaseEnv) -> torch.Tensor:
    penalty = torch.sum(torch.square(env.action_buffer._circular_buffer.buffer[:, -1, :] - env.action_buffer._circular_buffer.buffer[:, -2, :]), dim=1)
    return penalty


def zero_velocity_command_stillness_reward(
    env: BaseEnv,
    std: float,
    command_threshold: float = 1.0e-3,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward a quiet robot only while planar and yaw commands are zero."""
    from legged_lab.utils.stillness_reward import zero_velocity_command_stillness

    asset: Articulation = env.scene[asset_cfg.name]
    current_action = env.action_buffer._circular_buffer.buffer[:, -1, :]
    previous_action = env.action_buffer._circular_buffer.buffer[:, -2, :]
    return zero_velocity_command_stillness(
        asset.data.root_lin_vel_b,
        asset.data.root_ang_vel_b,
        asset.data.joint_vel,
        current_action,
        previous_action,
        env.command_generator.command,
        std=std,
        command_threshold=command_threshold,
    )


def zero_velocity_command_target_aligned_stillness_reward(
    env: BaseEnv,
    std: float,
    maximum_vector_angle: float,
    command_threshold: float = 1.0e-3,
    epsilon: float = 1.0e-8,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward stillness only after the full 3-D bar-vector is aligned."""

    from legged_lab.utils.stillness_reward import (
        gate_stillness_by_vector_alignment,
    )

    reward = zero_velocity_command_stillness_reward(
        env,
        std=std,
        command_threshold=command_threshold,
        asset_cfg=asset_cfg,
    )
    return gate_stillness_by_vector_alignment(
        reward,
        env.get_bar_vector_w(),
        env.get_target_vector_w(),
        maximum_angle=maximum_vector_angle,
        no_object_mask=getattr(env, "no_object_mask", None),
        epsilon=epsilon,
    )


def ankle_action_rate_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    current_actions = env.action_buffer._circular_buffer.buffer[:, -1, :]
    previous_actions = env.action_buffer._circular_buffer.buffer[:, -2, :]
    ankle_current = current_actions[:, asset_cfg.joint_ids]
    ankle_previous = previous_actions[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.square(ankle_current - ankle_previous), dim=1)
    return penalty


def undesired_contacts(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=1)


def fly(env: BaseEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    return torch.sum(is_contact, dim=-1) < 0.5


def flat_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    return reward


def is_terminated(env: BaseEnv) -> torch.Tensor:
    """Penalize terminated episodes that don't correspond to episodic timeouts."""
    return env.reset_buf * ~env.time_out_buf


def feet_air_time_positive_biped(
    env: BaseEnv,
    threshold: float,
    moving_command_threshold: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > moving_command_threshold
    return reward


def feet_slide(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > contact_force_threshold
    asset: Articulation = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    reward = torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)
    return reward


def body_force(env: BaseEnv, sensor_cfg: SceneEntityCfg, threshold: float = 500, max_reward: float = 400) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    reward = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].norm(dim=-1)
    reward[reward < threshold] = 0
    reward[reward > threshold] -= threshold
    reward = reward.clamp(min=0, max=max_reward)
    return reward


def feet_vertical_load_imbalance_penalty(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    contact_force_threshold: float = 25.0,
    command_threshold: float = 0.1,
    dead_zone: float = 0.1,
) -> torch.Tensor:
    """Penalize unequal mean vertical loads during commanded double support."""

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    vertical_force_history = contact_sensor.data.net_forces_w_history[
        :, :, sensor_cfg.body_ids, 2
    ]
    return normalized_two_foot_load_imbalance(
        vertical_force_history,
        env.command_generator.command,
        contact_force_threshold=contact_force_threshold,
        command_threshold=command_threshold,
        dead_zone=dead_zone,
    )


def joint_deviation_l1(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.abs(angle), dim=1)


def joint_position_l1_to_target(
    env: BaseEnv,
    target_position: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize absolute joint-position error from a fixed scalar target."""

    asset: Articulation = env.scene[asset_cfg.name]
    error = asset.data.joint_pos[:, asset_cfg.joint_ids] - target_position
    return torch.sum(torch.abs(error), dim=1)


def body_orientation_l2(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_orientation = math_utils.quat_rotate_inverse(asset.data.body_quat_w[:, asset_cfg.body_ids[0], :], asset.data.GRAVITY_VEC_W)
    reward = torch.sum(torch.square(body_orientation[:, :2]), dim=1)
    return reward


def feet_stumble(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    horizontal_vertical_force_ratio: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(torch.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2) > horizontal_vertical_force_ratio * torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2]), dim=1)


def feet_too_near_humanoid(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def knees_too_near_humanoid(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(knee_pos_w[:, 0] - knee_pos_w[:, 1], dim=-1)
    return (threshold - distance).clamp(min=0)


def feet_too_far_humanoid(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.36) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
    return (distance - threshold).clamp(min=0)


def knees_too_far_humanoid(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.34) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    knee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    distance = torch.norm(knee_pos_w[:, 0] - knee_pos_w[:, 1], dim=-1)
    return (distance - threshold).clamp(min=0)


def feet_orientation(env: BaseEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.2) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]

    feet_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
    x_axis = torch.tensor([1.0, 0.0, 0.0], device=feet_quat_w.device, dtype=feet_quat_w.dtype)
    x_axis = x_axis.unsqueeze(0).unsqueeze(1).expand(feet_quat_w.shape[0], 2, -1)
    feet_forward_dir_w = math_utils.quat_rotate(feet_quat_w, x_axis)
    feet_forward_dir_b = math_utils.quat_rotate_inverse(asset.data.root_quat_w.unsqueeze(1).expand(feet_forward_dir_w.shape[0], 2, -1), feet_forward_dir_w)

    env.feet_forward_dir_b = feet_forward_dir_b
    feet_y_component = torch.abs(feet_forward_dir_b[:, :, 1])
    penalty = torch.clamp(feet_y_component - threshold, min=0.0)

    return torch.sum(penalty, dim=1)


def upper_body_ee_left_quat_reward(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    left_quat_b = env.pose_command_generator.command[:, 0:4]
    left_quat_w = math_utils.quat_mul(asset.data.root_quat_w, left_quat_b)

    angle_error = math_utils.quat_error_magnitude(body_link_quat_w, left_quat_w)
    reward = torch.exp(-angle_error / std**2)
    return reward


def upper_body_ee_right_quat_reward(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    right_quat_b = env.pose_command_generator.command[:, 4:8]
    right_quat_w = math_utils.quat_mul(asset.data.root_quat_w, right_quat_b)

    angle_error = math_utils.quat_error_magnitude(body_link_quat_w, right_quat_w)
    reward = torch.exp(-angle_error / std**2)
    return reward


def loose_upper_body_ee_left_quat_reward(env: BaseEnv, std: float, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    left_quat_b = env.pose_command_generator.command[:, 0:4]
    left_quat_w = math_utils.quat_mul(asset.data.root_quat_w, left_quat_b)

    angle_error = math_utils.quat_error_magnitude(body_link_quat_w, left_quat_w)
    angle_error = torch.clamp(angle_error - deadband, min=0.0)
    reward = torch.exp(-angle_error / std**2)
    return reward


def loose_upper_body_ee_right_quat_reward(env: BaseEnv, std: float, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :].squeeze(1)
    right_quat_b = env.pose_command_generator.command[:, 4:8]
    right_quat_w = math_utils.quat_mul(asset.data.root_quat_w, right_quat_b)

    angle_error = math_utils.quat_error_magnitude(body_link_quat_w, right_quat_w)
    angle_error = torch.clamp(angle_error - deadband, min=0.0)
    reward = torch.exp(-angle_error / std**2)
    return reward


def upper_body_ee_left_xyz_quat_reward(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    left_xyz = env.pose_command_generator.command[:, 8:11]

    left_xyz_w = math_utils.quat_rotate(asset.data.root_quat_w, left_xyz)
    left_xyz_w = left_xyz_w + asset.data.root_pos_w
    distance = torch.norm(left_xyz_w - body_link_pos_w.squeeze(1), dim=1)
    return torch.exp(-distance / std**2)


def upper_body_ee_right_xyz_quat_reward(env: BaseEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    right_xyz = env.pose_command_generator.command[:, 11:14]

    right_xyz_w = math_utils.quat_rotate(asset.data.root_quat_w, right_xyz)
    right_xyz_w = right_xyz_w + asset.data.root_pos_w
    distance = torch.norm(right_xyz_w - body_link_pos_w.squeeze(1), dim=1)
    return torch.exp(-distance / std**2)


def loose_upper_body_ee_left_xyz_quat_reward(env: BaseEnv, std: float, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    left_xyz = env.pose_command_generator.command[:, 8:11]

    left_xyz_w = math_utils.quat_rotate(asset.data.root_quat_w, left_xyz)
    left_xyz_w = left_xyz_w + asset.data.root_pos_w
    distance = torch.norm(left_xyz_w - body_link_pos_w.squeeze(1), dim=1)
    distance = torch.clamp(distance - deadband, min=0.0)
    return torch.exp(-distance / std**2)


def loose_upper_body_ee_right_xyz_quat_reward(env: BaseEnv, std: float, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    body_link_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    right_xyz = env.pose_command_generator.command[:, 11:14]

    right_xyz_w = math_utils.quat_rotate(asset.data.root_quat_w, right_xyz)
    right_xyz_w = right_xyz_w + asset.data.root_pos_w
    distance = torch.norm(right_xyz_w - body_link_pos_w.squeeze(1), dim=1)
    distance = torch.clamp(distance - deadband, min=0.0)
    return torch.exp(-distance / std**2)


def loose_robot_height(env: BaseEnv, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height_command = env.command_generator.command[:, 3]

    root_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(root_height - height_command)
    loose_height_error = torch.clamp(height_error - deadband, min=0.0)
    return loose_height_error


def loose_robot_height_reward(env: BaseEnv, std: float, deadband: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    height_command = env.command_generator.command[:, 3]
    root_height = asset.data.root_pos_w[:, 2]
    height_error = torch.abs(root_height - height_command)
    loose_height_error = torch.clamp(height_error - deadband, min=0.0)
    reward = torch.exp(-loose_height_error / std**2)
    return reward


def penalty_feet_height(
    env: BaseEnv,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    contact_force_threshold: float,
    moving_command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.02,
) -> torch.Tensor:
    assert len(asset_cfg.body_ids) == 2
    asset: Articulation = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    feet_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]

    if not hasattr(env, "feet_height_rewarded"):
        env.feet_height_rewarded = torch.zeros((env.num_envs, 2), dtype=torch.bool, device=feet_height.device)

    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact_force_norm = torch.norm(contact_forces, dim=-1)
    contact = contact_force_norm > contact_force_threshold

    close_to_target = torch.abs(feet_height - target_height) < threshold
    should_reward = close_to_target & (~env.feet_height_rewarded) & (~contact)

    dif = torch.abs(feet_height - target_height)
    penalty = torch.where(env.feet_height_rewarded, torch.zeros_like(dif), dif)
    penalty = torch.min(penalty, dim=1).values

    env.feet_height_rewarded = (env.feet_height_rewarded | should_reward) & (~contact)

    penalty[env.command_generator.is_standing_env] = 0
    penalty *= (torch.norm(env.command_generator.command[:, :2], dim=1) + torch.abs(env.command_generator.command[:, 2])) > moving_command_threshold
    return torch.clip(penalty - threshold, min=0.)


def no_object_stand_anchor(env: BaseEnv, std: float = 0.5) -> torch.Tensor:
    """Penalize horizontal base speed in empty-hand environments."""
    if not hasattr(env, "no_object_mask"):
        return torch.zeros(env.num_envs, device=env.device)
    speed = torch.norm(env.robot.data.root_lin_vel_b[:, :2], dim=1)
    return speed * env.no_object_mask.float()
