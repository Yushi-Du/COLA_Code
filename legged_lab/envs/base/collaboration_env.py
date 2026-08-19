"""Runtime for fixed-bar phase-2 teachers and phase-3 students."""

from __future__ import annotations

import math
import os

import numpy as np
import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import EventManager, RewardManager
from isaaclab.managers.scene_entity_cfg import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.sim import PhysxCfg, SimulationContext
from isaaclab.utils.buffers import CircularBuffer, DelayBuffer
import isaacsim.core.utils.torch as torch_utils  # type: ignore
from rsl_rl.env import VecEnv

from legged_lab.mdp.commands import (
    UniformEEPoseCommandWorldFollowingQuatVel,
    UniformVelocityWorldHeightVelTargetVectorCommand,
)
from legged_lab.mdp.commands_cfg import (
    UniformEEPoseCommandWorldCfg,
    UniformVelocityHeightCommandCfg,
)
from legged_lab.utils.bar_controller import (
    BarController,
    target_vector_from_yaw,
    wrap_to_pi,
)
from legged_lab.utils.bar_geometry import (
    bar_vector_from_endpoints,
    centered_bar_teacher_observation,
    controller_force_effort,
    environment_relative_root_state,
    reset_human_effort_statistics,
)
from legged_lab.utils.env_utils.collaboration_scene import CollaborationSceneCfg


class FixedBarCollaborationEnv(VecEnv):
    def __init__(self, cfg, headless):
        self._distillation_mode = hasattr(cfg.robot, "student_obs_history_length")
        self.cfg = cfg
        self.headless = headless
        self.device = cfg.device
        self.physics_dt = cfg.sim.dt
        self.step_dt = cfg.sim.decimation * cfg.sim.dt
        self.num_envs = cfg.scene.num_envs
        self.seed(cfg.scene.seed)

        expected_dt = 1.0 / cfg.bar_controller.control_frequency_hz
        if not math.isclose(self.physics_dt, expected_dt, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "The centered-bar controllers must run once per physics step: "
                f"sim dt={self.physics_dt}, requested controller dt={expected_dt}."
            )

        sim_cfg = sim_utils.SimulationCfg(
            device=cfg.device,
            dt=cfg.sim.dt,
            render_interval=cfg.sim.decimation,
            log_dir=os.environ.get(
                "COLA_ISAACLAB_LOG_DIR", os.path.abspath("logs/isaaclab")
            ),
            physx=PhysxCfg(
                gpu_max_rigid_patch_count=cfg.sim.physx.gpu_max_rigid_patch_count,
                enable_external_forces_every_iteration=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )
        self.sim = SimulationContext(sim_cfg)
        scene_cfg = CollaborationSceneCfg(
            config=cfg.scene,
            bar_config=cfg.bar_controller,
            physics_dt=self.physics_dt,
            step_dt=self.step_dt,
            fixed_side=cfg.fixed_bar_side,
        )
        self.scene = InteractiveScene(scene_cfg)
        self.sim.reset()

        self.robot: Articulation = self.scene["robot"]
        self.carried_bar: RigidObject = self.scene["carried_bar"]
        self.contact_sensor: ContactSensor = self.scene.sensors["contact_sensor"]
        if cfg.scene.height_scanner.enable_height_scan:
            self.height_scanner: RayCaster = self.scene.sensors["height_scanner"]

        self.default_coms = None
        command_cfg = UniformVelocityHeightCommandCfg(
            asset_name="robot",
            resampling_time_range=cfg.commands.resampling_time_range,
            rel_standing_envs=cfg.commands.rel_standing_envs,
            rel_heading_envs=cfg.commands.rel_heading_envs,
            heading_command=cfg.commands.heading_command,
            heading_control_stiffness=cfg.commands.heading_control_stiffness,
            debug_vis=cfg.commands.debug_vis,
            ranges=cfg.commands.ranges,
        )
        pose_command_cfg = UniformEEPoseCommandWorldCfg(
            asset_name="robot",
            resampling_time_range=cfg.pose_commands.resampling_time_range,
            debug_vis=cfg.pose_commands.debug_vis,
            ranges=cfg.pose_commands.ranges,
        )
        self.command_generator = UniformVelocityWorldHeightVelTargetVectorCommand(
            cfg=command_cfg, env=self
        )
        self.pose_command_generator = UniformEEPoseCommandWorldFollowingQuatVel(
            cfg=pose_command_cfg, env=self
        )
        self.reward_manager = RewardManager(cfg.reward, self)

        self.init_buffers()
        self._initialize_controller_buffers()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.event_manager = EventManager(cfg.domain_rand.events, self)
        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")
        self.reset(env_ids)

    def init_buffers(self):
        self.extras = {}
        self.max_episode_length_s = self.cfg.scene.max_episode_length_s
        self.max_episode_length = np.ceil(self.max_episode_length_s / self.step_dt)
        self.num_actions = 29
        self.clip_actions = self.cfg.normalization.clip_actions
        self.clip_obs = self.cfg.normalization.clip_observations
        self.action_scale = self.cfg.robot.action_scale

        delay_cfg = self.cfg.domain_rand.action_delay
        self.action_buffer = DelayBuffer(
            delay_cfg.params["max_delay"], self.num_envs, device=self.device
        )
        self.action_buffer.compute(
            torch.zeros(self.num_envs, self.num_actions, device=self.device)
        )
        if delay_cfg.enable:
            self.action_buffer.set_time_lag(
                torch.randint(
                    delay_cfg.params["min_delay"],
                    delay_cfg.params["max_delay"] + 1,
                    (self.num_envs,),
                    dtype=torch.int,
                    device=self.device,
                ),
                torch.arange(self.num_envs, device=self.device),
            )

        self.robot_cfg = SceneEntityCfg("robot")
        self.robot_cfg.resolve(self.scene)
        self.termination_contact_cfg = SceneEntityCfg(
            "contact_sensor",
            body_names=self.cfg.robot.terminate_contacts_body_names,
        )
        self.termination_contact_cfg.resolve(self.scene)
        self.feet_cfg = SceneEntityCfg(
            "contact_sensor", body_names=self.cfg.robot.feet_body_names
        )
        self.feet_cfg.resolve(self.scene)

        self.obs_scales = self.cfg.normalization.obs_scales
        self.add_noise = self.cfg.noise.add_noise
        self.episode_length_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )
        self.sim_step_counter = 0
        self.time_out_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.init_obs_buffer()

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_scales = self.cfg.noise.noise_scales
            self.noise_scale_vec = torch.zeros_like(actor_obs[0])
            self.noise_scale_vec[18:21] = (
                noise_scales.ang_vel * self.obs_scales.ang_vel
            )
            self.noise_scale_vec[21:24] = (
                noise_scales.projected_gravity * self.obs_scales.projected_gravity
            )
            self.noise_scale_vec[24:53] = (
                noise_scales.joint_pos * self.obs_scales.joint_pos
            )
            self.noise_scale_vec[53:82] = (
                noise_scales.joint_vel * self.obs_scales.joint_vel
            )
            if self.cfg.scene.height_scanner.enable_height_scan:
                height_scan = (
                    self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                    - self.height_scanner.data.ray_hits_w[..., 2]
                    - self.cfg.normalization.height_scan_offset
                )
                self.height_scan_noise_vec = torch.full_like(
                    height_scan[0],
                    noise_scales.height_scan * self.obs_scales.height_scan,
                )

        if self._distillation_mode:
            actor_history = self.cfg.robot.student_obs_history_length
            privileged_history = self.cfg.robot.teacher_obs_history_length
        else:
            actor_history = self.cfg.robot.actor_obs_history_length
            privileged_history = self.cfg.robot.critic_obs_history_length

        self.actor_obs_buffer = CircularBuffer(
            max_len=actor_history,
            batch_size=self.num_envs,
            device=self.device,
        )
        privileged_buffer = CircularBuffer(
            max_len=privileged_history,
            batch_size=self.num_envs,
            device=self.device,
        )
        if self._distillation_mode:
            self.teacher_obs_buffer = privileged_buffer
        else:
            self.critic_obs_buffer = privileged_buffer

    def _initialize_controller_buffers(self):
        n = self.num_envs
        self.no_object_mask = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.controller_requested_height_w = torch.zeros(n, device=self.device)
        self.controller_requested_velocity_xy = torch.zeros(n, 2, device=self.device)
        self.human_effort_force_time_integral = torch.zeros(n, device=self.device)
        self.human_effort_elapsed_time = torch.zeros(n, device=self.device)
        self._bar_mass = torch.full(
            (n,), self.cfg.bar_controller.bar_mass, device=self.device
        )

        cfg = self.cfg.bar_controller
        self.bar_controller = BarController(
            cfg,
            num_envs=self.num_envs,
            device=self.device,
            dt=self.physics_dt,
        )
        controller = self.bar_controller
        self.controller_force_w = controller.force_w
        self.controller_torque_w = controller.torque_w
        self.controller_requested_yaw_w = torch.full(
            (n,),
            cfg.default_target_yaw_w,
            device=self.device,
            dtype=torch.float32,
        )
        self.controller_reference_yaw_w = controller.yaw_reference_w
        self.controller_measured_yaw_w = controller.current_bar_yaw_w
        self.controller_measured_yaw_rate_w = controller.current_bar_yaw_rate_w
        self.controller_yaw_error = controller.yaw_error
        self.controller_yaw_axis_w = controller.yaw_axis_w
        self.controller_yaw_scalar_torque = controller.yaw_control_torque

        contract = getattr(self, "_controller_force_jitter_contract", None)
        if contract is not None:
            history_length, _ = contract
            self.controller_force_jitter_history_w = torch.zeros(
                n,
                history_length,
                3,
                device=self.device,
                dtype=self.controller_force_w.dtype,
            )
            self.controller_force_jitter_valid_samples = torch.zeros(
                n, dtype=torch.long, device=self.device
            )
            self.controller_force_jitter_cursor = 0

    def get_human_endpoint_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.cfg.bar_controller
        local_offset = torch.tensor(
            cfg.human_endpoint_offset, device=self.device, dtype=torch.float32
        ).repeat(self.num_envs, 1)
        root_quat = self.carried_bar.data.root_quat_w
        endpoint_position = self.carried_bar.data.root_pos_w + math_utils.quat_apply(
            root_quat, local_offset
        )
        com_offset = endpoint_position - self.carried_bar.data.root_com_pos_w
        endpoint_velocity = self.carried_bar.data.root_com_lin_vel_w + torch.linalg.cross(
            self.carried_bar.data.root_com_ang_vel_w, com_offset, dim=-1
        )
        return endpoint_position, endpoint_velocity

    def get_robot_endpoint_position(self) -> torch.Tensor:
        local_offset = torch.tensor(
            self.cfg.bar_controller.robot_endpoint_offset,
            device=self.device,
            dtype=torch.float32,
        ).repeat(self.num_envs, 1)
        return self.carried_bar.data.root_pos_w + math_utils.quat_apply(
            self.carried_bar.data.root_quat_w, local_offset
        )

    def get_bar_vector_w(self) -> torch.Tensor:
        """Return the current world-frame robot-end to human-end bar vector."""

        return bar_vector_from_endpoints(
            self.get_robot_endpoint_position(),
            self.get_human_endpoint_state()[0],
        )

    def _refresh_bar_mass_properties(self):
        masses = self.carried_bar.root_physx_view.get_masses().to(self.device)
        self._bar_mass.copy_(masses[:, 0])

    def _terrain_relative_height_target(self) -> torch.Tensor:
        """Convert the sampled link-height command to endpoint world height."""

        target = self.command_generator.link_height_command_b[:, 0].clone()
        ankle_ids, _ = self.robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"]
        )
        ankle_heights = self.robot.data.body_pos_w[:, ankle_ids, 2]
        height_cfg = self.cfg.support_height_control
        lower_ankle_height = torch.min(ankle_heights, dim=1).values + height_cfg.ankle_height_offset
        old_support_root_height = self.scene.env_origins[:, 2] + 0.14
        terrain_difference = lower_ankle_height - old_support_root_height
        correction_mask = (
            torch.abs(terrain_difference) > height_cfg.ankle_height_correction_threshold
        )
        target[correction_mask] += terrain_difference[correction_mask]
        target.clamp_(*height_cfg.corrected_height_range)
        return target + self.scene.env_origins[:, 2]

    def configure_controller_force_jitter_tracking(
        self, history_length: int, settled_tolerance: float
    ):
        """Register the shared 400 Hz force-history contract for reward terms."""

        contract = (int(history_length), float(settled_tolerance))
        if contract[0] < 3:
            raise ValueError("force-jitter history_length must be at least three")
        existing = getattr(self, "_controller_force_jitter_contract", None)
        if existing is not None and existing != contract:
            raise ValueError(
                "All controller force-jitter rewards must use one history contract: "
                f"existing={existing}, requested={contract}."
            )
        self._controller_force_jitter_contract = contract

    def get_controller_force_jitter_history(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return chronologically ordered 400 Hz force history and valid counts."""

        history = torch.roll(
            self.controller_force_jitter_history_w,
            shifts=-self.controller_force_jitter_cursor,
            dims=1,
        )
        return history, self.controller_force_jitter_valid_samples

    def _record_controller_force_jitter_sample(self):
        contract = getattr(self, "_controller_force_jitter_contract", None)
        if contract is None:
            return
        history_length, settled_tolerance = contract
        controller = self.bar_controller
        height_settled = torch.abs(
            self.controller_requested_height_w - controller.height_reference
        ) <= settled_tolerance
        velocity_settled = torch.all(
            torch.abs(
                self.controller_requested_velocity_xy
                - controller.velocity_reference_xy
            )
            <= settled_tolerance,
            dim=1,
        )
        yaw_settled = torch.abs(
            wrap_to_pi(
                self.controller_requested_yaw_w - controller.yaw_reference_w
            )
        ) <= settled_tolerance
        settled = (
            height_settled
            & velocity_settled
            & yaw_settled
            & (~self.no_object_mask)
        )

        self.controller_force_jitter_history_w[
            :, self.controller_force_jitter_cursor
        ].copy_(self.controller_force_w)
        self.controller_force_jitter_valid_samples.copy_(
            torch.where(
                settled,
                torch.clamp(
                    self.controller_force_jitter_valid_samples + 1,
                    max=history_length,
                ),
                torch.zeros_like(self.controller_force_jitter_valid_samples),
            )
        )
        self.controller_force_jitter_cursor = (
            self.controller_force_jitter_cursor + 1
        ) % history_length

    def get_controller_point_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the bar center-of-mass position and velocity in world frame."""

        return (
            self.carried_bar.data.root_com_pos_w,
            self.carried_bar.data.root_com_lin_vel_w,
        )

    def get_target_vector_w(self) -> torch.Tensor:
        """Return the requested unit bar-vector in the world frame."""

        return target_vector_from_yaw(self.controller_requested_yaw_w)

    def _privileged_bar_observation(self) -> torch.Tensor:
        """Append mass and all three controller commands for the teacher."""

        object_state = environment_relative_root_state(
            self.carried_bar.data.root_state_w,
            self.scene.env_origins,
        )
        requested_height_local = (
            self._terrain_relative_height_target() - self.scene.env_origins[:, 2]
        ).unsqueeze(1)
        requested_velocity_w = self.command_generator.vel_command_w[:, :2]
        target_vector_w = self.command_generator.target_vector_w
        # ``init_obs_buffer`` asks for one observation before controller buffers
        # and the randomized PhysX mass cache are created. This pass is used
        # only to size the noise vector; reset refreshes ``_bar_mass`` before
        # the first observation returned to training.
        object_mass = getattr(self, "_bar_mass", None)
        if object_mass is None:
            object_mass = torch.full(
                (self.num_envs,),
                self.cfg.bar_controller.bar_mass,
                device=self.device,
            )
        return centered_bar_teacher_observation(
            object_state,
            object_mass.unsqueeze(1),
            requested_height_local,
            requested_velocity_w,
            target_vector_w,
        )

    def _reset_controller(self, env_ids: torch.Tensor):
        center_position, center_velocity = self.get_controller_point_state()
        controller_cfg = self.cfg.bar_controller
        self.bar_controller.reset_gains(
            env_ids,
            randomize=controller_cfg.gain_randomization_enabled,
            scale_range=controller_cfg.gain_randomization_scale_range,
        )
        if hasattr(self.command_generator, "target_vector_yaw_w"):
            self.controller_requested_yaw_w[env_ids] = (
                self.command_generator.target_vector_yaw_w[env_ids]
            )
        else:
            self.controller_requested_yaw_w[env_ids] = (
                controller_cfg.default_target_yaw_w
            )
        self.bar_controller.reset(
            env_ids,
            center_position_w=center_position,
            center_velocity_w=center_velocity,
            bar_vector_w=self.get_bar_vector_w(),
            angular_velocity_w=self.carried_bar.data.root_com_ang_vel_w,
        )
        self.controller_requested_height_w[env_ids] = center_position[env_ids, 2]
        self.controller_requested_velocity_xy[env_ids] = center_velocity[env_ids, :2]
        reset_human_effort_statistics(
            self.human_effort_force_time_integral,
            self.human_effort_elapsed_time,
            env_ids,
        )
        if hasattr(self, "controller_force_jitter_valid_samples"):
            self.controller_force_jitter_history_w[env_ids] = 0.0
            self.controller_force_jitter_valid_samples[env_ids] = 0

    def _update_endpoint_controller(self):
        """Evaluate and stage the centered world-frame force and yaw torque."""

        motion_cfg = self.cfg.experiment.support_motion
        center_position, center_velocity = self.get_controller_point_state()

        min_height, max_height = motion_cfg.bar_height_command_range
        self.command_generator.height_command_b[:, 0] = torch.clamp(
            center_position[:, 2] + motion_cfg.bar_height_command_offset,
            min_height,
            max_height,
        )
        if self.no_object_mask.any():
            self.command_generator.vel_command_w[self.no_object_mask] = 0.0
            self.command_generator.height_command_b[
                self.no_object_mask, 0
            ] = motion_cfg.no_object_height_command

        requested_height = self._terrain_relative_height_target()
        requested_velocity = self.command_generator.vel_command_w[:, :2].clone()
        self.controller_requested_height_w.copy_(requested_height)
        self.controller_requested_velocity_xy.copy_(requested_velocity)
        if hasattr(self.command_generator, "target_vector_yaw_w"):
            self.controller_requested_yaw_w.copy_(
                self.command_generator.target_vector_yaw_w
            )

        self.bar_controller.compute(
            requested_height_w=requested_height,
            requested_velocity_xy_w=requested_velocity,
            requested_yaw_w=self.controller_requested_yaw_w,
            center_position_w=center_position,
            center_velocity_w=center_velocity,
            bar_vector_w=self.get_bar_vector_w(),
            angular_velocity_w=self.carried_bar.data.root_com_ang_vel_w,
            bar_mass=self._bar_mass,
        )
        self.controller_force_w[self.no_object_mask] = 0.0
        self.controller_torque_w[self.no_object_mask] = 0.0
        self._record_controller_force_jitter_sample()

        active_bar = (~self.no_object_mask).to(self.controller_force_w.dtype)
        instantaneous_effort = controller_force_effort(self.controller_force_w)
        self.human_effort_force_time_integral += (
            instantaneous_effort * active_bar * self.physics_dt
        )
        self.human_effort_elapsed_time += active_bar * self.physics_dt

        wrench_composer = self.carried_bar.instantaneous_wrench_composer
        wrench_composer.set_forces_and_torques(
            forces=self.controller_force_w.unsqueeze(1),
            positions=center_position.unsqueeze(1),
            is_global=True,
        )
        # Isaac Lab 2.3's set kernel overwrites an explicit torque when a force
        # position is also supplied. Add the independent torque afterwards so
        # the final body-frame wrench contains both r x F and the yaw torque.
        wrench_composer.add_forces_and_torques(
            torques=self.controller_torque_w.unsqueeze(1),
            is_global=True,
        )

        callback = getattr(self, "_centered_controller_substep_callback", None)
        if callback is not None:
            callback()

    def compute_current_observations(self):
        robot = self.robot
        command = self.command_generator.command.clone()
        pose_command = self.pose_command_generator.command
        angular_velocity = robot.data.root_ang_vel_b
        projected_gravity = robot.data.projected_gravity_b
        joint_position = (robot.data.joint_pos - robot.data.default_joint_pos)[:, :29]
        joint_velocity = (robot.data.joint_vel - robot.data.default_joint_vel)[:, :29]
        action = self.action_buffer._circular_buffer.buffer[:, -1, :]

        command_w = torch.zeros(self.num_envs, 3, device=self.device)
        command_w[:, :2] = command[:, :2]
        command_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(robot.data.root_quat_w), command_w
        )
        command = torch.cat((command_b[:, :2], command[:, 2:4]), dim=-1)

        # Canonical 13-D rigid-object state. Position is relative to the
        # corresponding terrain environment origin so identical local scenes
        # have identical observations. Quaternion and velocities remain in
        # the world frame, matching Isaac Lab's root-state convention.
        object_observation = self._privileged_bar_observation()

        masked_command = command.clone()
        if self.cfg.experiment.observations.mask_planar_velocity_command:
            masked_command[:, :3] = 0.0

        proprioception = (
            pose_command,
            angular_velocity * self.obs_scales.ang_vel,
            projected_gravity * self.obs_scales.projected_gravity,
            joint_position * self.obs_scales.joint_pos,
            joint_velocity * self.obs_scales.joint_vel,
            action * self.obs_scales.actions,
        )

        if self._distillation_mode:
            student_command = masked_command.clone()
            masked_height = self.cfg.experiment.observations.masked_height_command
            if masked_height is not None:
                student_command[:, 3] = masked_height
            actor_obs = torch.cat(
                (student_command * self.obs_scales.commands, *proprioception), dim=-1
            )
            teacher_obs = torch.cat(
                (
                    masked_command * self.obs_scales.commands,
                    *proprioception,
                    object_observation,
                ),
                dim=-1,
            )
            return actor_obs, teacher_obs

        actor_obs = torch.cat(
            (
                masked_command * self.obs_scales.commands,
                *proprioception,
                object_observation,
            ),
            dim=-1,
        )
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        feet_contact = (
            torch.max(
                torch.linalg.vector_norm(
                    net_contact_forces[:, :, self.feet_cfg.body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > self.cfg.experiment.termination_observation.feet_contact_force_threshold
        )
        critic_obs = torch.cat(
            (
                actor_obs,
                robot.data.root_lin_vel_b * self.obs_scales.lin_vel,
                feet_contact,
            ),
            dim=-1,
        )
        return actor_obs, critic_obs

    def compute_observations(self):
        current_actor_obs, current_privileged_obs = self.compute_current_observations()
        if self.add_noise:
            current_actor_obs += (2 * torch.rand_like(current_actor_obs) - 1) * self.noise_scale_vec
        self.actor_obs_buffer.append(current_actor_obs)
        if self._distillation_mode:
            self.teacher_obs_buffer.append(current_privileged_obs)
            privileged_obs = self.teacher_obs_buffer.buffer.reshape(self.num_envs, -1)
        else:
            self.critic_obs_buffer.append(current_privileged_obs)
            privileged_obs = self.critic_obs_buffer.buffer.reshape(self.num_envs, -1)
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)

        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            privileged_obs = torch.cat((privileged_obs, height_scan), dim=-1)
            if self.add_noise:
                height_scan += (2 * torch.rand_like(height_scan) - 1) * self.height_scan_noise_vec
            actor_obs = torch.cat((actor_obs, height_scan), dim=-1)

        return (
            torch.clamp(actor_obs, -self.clip_obs, self.clip_obs),
            torch.clamp(privileged_obs, -self.clip_obs, self.clip_obs),
        )

    def reset(self, env_ids):
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self.extras["log"] = {}
        if self.cfg.scene.terrain_generator is not None and self.cfg.scene.terrain_generator.curriculum:
            self.extras["log"].update(self.update_terrain_levels(ids))

        self.scene.reset(ids)
        if "reset" in self.event_manager.available_modes:
            self.event_manager.apply(
                mode="reset",
                env_ids=ids,
                dt=self.step_dt,
                global_env_step_count=self.sim_step_counter // self.cfg.sim.decimation,
            )
        self._refresh_bar_mass_properties()

        self.extras["log"].update(self.reward_manager.reset(ids))
        self.extras["time_outs"] = self.time_out_buf
        self.command_generator.reset(ids)
        self.pose_command_generator.reset(ids)
        self.actor_obs_buffer.reset(ids)
        if self._distillation_mode:
            self.teacher_obs_buffer.reset(ids)
        else:
            self.critic_obs_buffer.reset(ids)
        self.action_buffer.reset(ids)
        self.episode_length_buf[ids] = 0

        self.no_object_mask[ids] = (
            torch.rand(len(ids), device=self.device)
            < self.cfg.experiment.support_motion.no_object_probability
        )
        park_ids = ids[self.no_object_mask[ids]]
        if len(park_ids) > 0:
            park_offset = torch.tensor(
                self.cfg.bar_controller.no_object_park_offset,
                device=self.device,
                dtype=torch.float32,
            ).repeat(len(park_ids), 1)
            park_position = self.robot.data.root_pos_w[park_ids] + math_utils.quat_apply(
                self.robot.data.root_quat_w[park_ids], park_offset
            )
            park_orientation = self.robot.data.root_quat_w[park_ids]
            self.carried_bar.write_root_pose_to_sim(
                torch.cat((park_position, park_orientation), dim=-1), env_ids=park_ids
            )
            self.carried_bar.write_root_velocity_to_sim(
                torch.zeros(len(park_ids), 6, device=self.device), env_ids=park_ids
            )
            self.command_generator.vel_command_w[park_ids] = 0.0
            self.command_generator.height_command_b[
                park_ids, 0
            ] = self.cfg.experiment.support_motion.no_object_height_command

        self._reset_controller(ids)
        self.scene.write_data_to_sim()
        self.sim.forward()

    def step(self, actions: torch.Tensor):
        delayed_actions = self.action_buffer.compute(actions)
        clipped_actions = torch.clamp(
            delayed_actions, -self.clip_actions, self.clip_actions
        ).to(self.device)
        processed_actions = clipped_actions * self.action_scale + self.robot.data.default_joint_pos

        # One controller update per 2.5 ms physics substep: exactly 400 Hz.
        for _ in range(self.cfg.sim.decimation):
            self.sim_step_counter += 1
            self.robot.set_joint_position_target(processed_actions)
            self._update_endpoint_controller()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.scene.update(dt=self.physics_dt)

        if not self.headless:
            self.sim.render()

        self.episode_length_buf += 1
        self.command_generator.compute(self.step_dt)
        self.pose_command_generator.compute(self.step_dt)
        if self.no_object_mask.any():
            self.command_generator.vel_command_w[self.no_object_mask] = 0.0
            self.command_generator.height_command_b[
                self.no_object_mask, 0
            ] = self.cfg.experiment.support_motion.no_object_height_command
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)

        self.reset_buf, self.time_out_buf = self.check_reset()
        reward_buf = self.reward_manager.compute(self.step_dt)
        self.reset(self.reset_buf.nonzero(as_tuple=False).flatten())

        actor_obs, privileged_obs = self.compute_observations()
        privileged_key = "teacher" if self._distillation_mode else "critic"
        self.extras["observations"] = {privileged_key: privileged_obs}
        return actor_obs, reward_buf, self.reset_buf, self.extras

    def check_reset(self):
        net_contact_forces = self.contact_sensor.data.net_forces_w_history
        ankle_ids, _ = self.robot.find_bodies(
            ["left_ankle_roll_link", "right_ankle_roll_link"]
        )
        lower_ankle_height = (
            torch.min(self.robot.data.body_pos_w[:, ankle_ids, 2], dim=1).values
            + self.cfg.support_height_control.ankle_height_offset
        )
        human_position = self.get_human_endpoint_state()[0]
        robot_position = self.get_robot_endpoint_position()
        minimum_height = self.cfg.experiment.termination_observation.object_min_height
        object_low = (human_position[:, 2] - lower_ankle_height < minimum_height) | (
            robot_position[:, 2] - lower_ankle_height < minimum_height
        )
        object_low &= ~self.no_object_mask

        threshold = self.cfg.experiment.termination_observation.robot_contact_force_threshold
        reset_buf = torch.any(
            torch.max(
                torch.linalg.vector_norm(
                    net_contact_forces[:, :, self.termination_contact_cfg.body_ids], dim=-1
                ),
                dim=1,
            )[0]
            > threshold,
            dim=1,
        )
        time_out_buf = self.episode_length_buf >= self.max_episode_length
        return reset_buf | time_out_buf | object_low, time_out_buf

    def get_observations(self):
        actor_obs, privileged_obs = self.compute_observations()
        privileged_key = "teacher" if self._distillation_mode else "critic"
        self.extras["observations"] = {privileged_key: privileged_obs}
        return actor_obs, self.extras

    def update_terrain_levels(self, env_ids):
        distance = torch.linalg.vector_norm(
            self.robot.data.root_pos_w[env_ids, :2]
            - self.scene.env_origins[env_ids, :2],
            dim=1,
        )
        move_up = distance > self.scene.terrain.cfg.terrain_generator.size[0] / 2
        move_down = distance < (
            torch.linalg.vector_norm(
                self.command_generator.command[env_ids, :2], dim=1
            )
            * self.max_episode_length_s
            * 0.5
        )
        move_down &= ~move_up
        self.scene.terrain.update_env_origins(env_ids, move_up, move_down)
        return {
            "Curriculum/terrain_levels": torch.mean(
                self.scene.terrain.terrain_levels.float()
            )
        }

    @staticmethod
    def seed(seed: int = -1) -> int:
        try:
            import omni.replicator.core as rep  # type: ignore

            rep.set_global_seed(seed)
        except ModuleNotFoundError:
            pass
        return torch_utils.set_seed(seed)


__all__ = ["FixedBarCollaborationEnv"]
