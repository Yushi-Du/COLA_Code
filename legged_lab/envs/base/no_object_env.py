"""Runtime for the fixed-hand collaboration population without an object."""

from __future__ import annotations

import os

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sim import PhysxCfg
from isaaclab.sim.utils import get_current_stage
from isaaclab.utils.buffers import CircularBuffer

from legged_lab.envs.base.locomotion_env import LocomotionEnv
from legged_lab.mdp.commands import UniformEEPoseCommandWorldFollowingQuatVel
from legged_lab.mdp.commands_cfg import UniformEEPoseCommandWorldCfg
from legged_lab.utils.env_utils.no_object_scene import NoObjectSceneCfg
from legged_lab.utils.mass_observation import no_object_mass_observation
from legged_lab.utils.static_population import (
    NO_OBJECT_PRIVILEGED_DIM,
    NO_OBJECT_TOPOLOGY_ID,
    static_no_object_privileged_tail,
)


class NoObjectEnv(LocomotionEnv):
    """Make a pure-G1 scene observation-compatible with fixed-middle COLA."""

    def __init__(self, cfg, headless):
        self._distillation_mode = hasattr(cfg.robot, "student_obs_history_length")
        super().__init__(cfg, headless)
        # The distillation runner routes this static population through the frozen
        # locomotion teacher instead of the collaboration residual teacher.
        # Every sample on this rank is physically object-free.
        self.teacher_locomotion_mask = torch.ones(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.cola_topology_id = NO_OBJECT_TOPOLOGY_ID
        self.cola_topology_name = "no_object_fixed_hand"
        self.static_no_object_population = True
        if "carried_bar" in self.scene.rigid_objects:
            raise RuntimeError("static no-object scene unexpectedly contains a bar")
        if "fixed_bar_attachments" in self.scene.extras:
            raise RuntimeError(
                "static no-object scene unexpectedly contains wrist constraints"
            )
        stage = get_current_stage()
        attachment_roots = (
            f"{self.scene.env_ns}/env_0/FixedBarAttachments",
        )
        if any(stage.GetPrimAtPath(path).IsValid() for path in attachment_roots):
            raise RuntimeError(
                "static no-object stage unexpectedly contains wrist-attachment prims"
            )

    def _create_simulation_cfg(self):
        """Match fixed-middle contact material while omitting object-force support."""

        return sim_utils.SimulationCfg(
            device=self.cfg.device,
            dt=self.cfg.sim.dt,
            render_interval=self.cfg.sim.decimation,
            log_dir=os.environ.get(
                "COLA_ISAACLAB_LOG_DIR", os.path.abspath("logs/isaaclab")
            ),
            physx=PhysxCfg(
                gpu_max_rigid_patch_count=(
                    self.cfg.sim.physx.gpu_max_rigid_patch_count
                ),
                enable_external_forces_every_iteration=True,
            ),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
            ),
        )

    def _create_scene_cfg(self):
        return NoObjectSceneCfg(
            config=self.cfg.scene,
            physics_dt=self.physics_dt,
            step_dt=self.step_dt,
        )

    def _create_pose_command_generator(self):
        pose_command_cfg = UniformEEPoseCommandWorldCfg(
            asset_name="robot",
            resampling_time_range=self.cfg.pose_commands.resampling_time_range,
            debug_vis=self.cfg.pose_commands.debug_vis,
            ranges=self.cfg.pose_commands.ranges,
        )
        return UniformEEPoseCommandWorldFollowingQuatVel(
            cfg=pose_command_cfg,
            env=self,
        )

    def _fixed_middle_command(self) -> torch.Tensor:
        """Return the four-D command in the fixed-middle observation convention."""

        command = self.command_generator.command.clone()
        command_w = torch.zeros(self.num_envs, 3, device=self.device)
        command_w[:, :2] = command[:, :2]
        command_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(self.robot.data.root_quat_w), command_w
        )
        command = torch.cat((command_b[:, :2], command[:, 2:4]), dim=1)
        if self.cfg.experiment.observations.mask_planar_velocity_command:
            command[:, :3] = 0.0
        return command

    def _student_mass_observation_enabled(self) -> bool:
        observations_cfg = self.cfg.experiment.observations
        return bool(
            getattr(observations_cfg, "student_mass_observation_enabled", False)
        )

    def _ensure_student_mass_observation_buffers(self):
        if hasattr(self, "_student_mass_observation_kg"):
            return
        observations_cfg = self.cfg.experiment.observations
        low, high = observations_cfg.student_no_object_true_mass_range_kg
        midpoint = 0.5 * (float(low) + float(high))
        self._student_mass_pseudo_true_kg = torch.full(
            (self.num_envs,), midpoint, device=self.device, dtype=torch.float32
        )
        self._student_mass_bias_kg = torch.zeros_like(
            self._student_mass_pseudo_true_kg
        )
        self._student_mass_observation_kg = self._student_mass_pseudo_true_kg.clone()

    def _student_mass_observation(self) -> torch.Tensor:
        self._ensure_student_mass_observation_buffers()
        return self._student_mass_observation_kg

    def _reset_student_mass_observation(self, env_ids: torch.Tensor):
        if not self._student_mass_observation_enabled():
            return
        self._ensure_student_mass_observation_buffers()
        reference = self._student_mass_observation_kg[env_ids]
        observation_kg, pseudo_true_kg, bias_kg = no_object_mass_observation(
            reference,
            self.cfg.experiment.observations.student_no_object_true_mass_range_kg,
            self.cfg.experiment.observations.student_mass_bias_range_kg,
        )
        self._student_mass_pseudo_true_kg[env_ids] = pseudo_true_kg
        self._student_mass_bias_kg[env_ids] = bias_kg
        self._student_mass_observation_kg[env_ids] = observation_kg

    def reset(self, env_ids):
        super().reset(env_ids)
        if len(env_ids) == 0:
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_student_mass_observation(ids)

    def _base_proprioception(self) -> tuple[torch.Tensor, ...]:
        robot = self.robot
        return (
            self.pose_command_generator.command,
            robot.data.root_ang_vel_b * self.obs_scales.ang_vel,
            robot.data.projected_gravity_b * self.obs_scales.projected_gravity,
            (robot.data.joint_pos - robot.data.default_joint_pos)[:, :29]
            * self.obs_scales.joint_pos,
            (robot.data.joint_vel - robot.data.default_joint_vel)[:, :29]
            * self.obs_scales.joint_vel,
            self.action_buffer._circular_buffer.buffer[:, -1, :]
            * self.obs_scales.actions,
        )

    def _no_object_tail(self, command: torch.Tensor) -> torch.Tensor:
        tail = static_no_object_privileged_tail(
            self.num_envs,
            self.device,
            height_command=command[:, 3:4],
            dtype=command.dtype,
        )
        if tail.shape != (self.num_envs, NO_OBJECT_PRIVILEGED_DIM):
            raise RuntimeError("invalid static no-object teacher tail")
        return tail

    def compute_current_observations(self):
        command = self._fixed_middle_command()
        proprioception = self._base_proprioception()
        privileged_tail = self._no_object_tail(command)

        if self._distillation_mode:
            student_command = command.clone()
            masked_height = self.cfg.experiment.observations.masked_height_command
            if masked_height is not None:
                student_command[:, 3] = masked_height
            student_features = (
                (self._student_mass_observation().unsqueeze(1),)
                if self._student_mass_observation_enabled()
                else ()
            )
            actor_obs = torch.cat(
                (
                    student_command * self.obs_scales.commands,
                    *proprioception,
                    *student_features,
                ),
                dim=1,
            )
            teacher_obs = torch.cat(
                (
                    command * self.obs_scales.commands,
                    *proprioception,
                    privileged_tail,
                ),
                dim=1,
            )
            return actor_obs, teacher_obs

        actor_obs = torch.cat(
            (
                command * self.obs_scales.commands,
                *proprioception,
                privileged_tail,
            ),
            dim=1,
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
                self.robot.data.root_lin_vel_b * self.obs_scales.lin_vel,
                feet_contact,
            ),
            dim=1,
        )
        return actor_obs, critic_obs

    def init_obs_buffer(self):
        if self.add_noise:
            actor_obs, _ = self.compute_current_observations()
            noise_vec = torch.zeros_like(actor_obs[0])
            noise_scales = self.cfg.noise.noise_scales
            noise_vec[18:21] = noise_scales.ang_vel * self.obs_scales.ang_vel
            noise_vec[21:24] = (
                noise_scales.projected_gravity * self.obs_scales.projected_gravity
            )
            noise_vec[24:53] = noise_scales.joint_pos * self.obs_scales.joint_pos
            noise_vec[53:82] = noise_scales.joint_vel * self.obs_scales.joint_vel
            self.noise_scale_vec = noise_vec

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
            self.critic_obs_buffer = self.teacher_obs_buffer
        else:
            self.critic_obs_buffer = privileged_buffer

    def compute_observations(self):
        actor_frame, privileged_frame = self.compute_current_observations()
        if self.add_noise:
            actor_frame += (
                2 * torch.rand_like(actor_frame) - 1
            ) * self.noise_scale_vec
        self.actor_obs_buffer.append(actor_frame)
        if self._distillation_mode:
            self.teacher_obs_buffer.append(privileged_frame)
            privileged_obs = self.teacher_obs_buffer.buffer.reshape(
                self.num_envs, -1
            )
        else:
            self.critic_obs_buffer.append(privileged_frame)
            privileged_obs = self.critic_obs_buffer.buffer.reshape(
                self.num_envs, -1
            )
        actor_obs = self.actor_obs_buffer.buffer.reshape(self.num_envs, -1)

        if self.cfg.scene.height_scanner.enable_height_scan:
            height_scan = (
                self.height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self.height_scanner.data.ray_hits_w[..., 2]
                - self.cfg.normalization.height_scan_offset
            ) * self.obs_scales.height_scan
            privileged_obs = torch.cat((privileged_obs, height_scan), dim=1)
            if self.add_noise:
                height_scan += (
                    2 * torch.rand_like(height_scan) - 1
                ) * self.height_scan_noise_vec
            actor_obs = torch.cat((actor_obs, height_scan), dim=1)

        return (
            torch.clamp(actor_obs, -self.clip_obs, self.clip_obs),
            torch.clamp(privileged_obs, -self.clip_obs, self.clip_obs),
        )

    def get_observations(self):
        actor_obs, privileged_obs = self.compute_observations()
        key = "teacher" if self._distillation_mode else "critic"
        self.extras["observations"] = {key: privileged_obs}
        return actor_obs, self.extras

    def step(self, actions: torch.Tensor):
        actor_obs, rewards, resets, extras = super().step(actions)
        if self._distillation_mode:
            teacher_obs = extras["observations"].pop("critic")
            extras["observations"]["teacher"] = teacher_obs
        return actor_obs, rewards, resets, extras



__all__ = ["NoObjectEnv"]
