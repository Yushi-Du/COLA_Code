#!/usr/bin/env python3
"""Run a COLA policy in the robot-only MuJoCo scene."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import time

import mujoco
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "model" / "scene.xml"
DEFAULT_POLICY = ROOT / "student.jit"

SIMULATION_DT = 0.001
POLICY_DT = 0.02
POLICY_DECIMATION = round(POLICY_DT / SIMULATION_DT)
NUM_ACTIONS = 29
OBSERVATION_FRAME_SIZE = 111
PHASE1_OBSERVATION_HISTORY = 10
STUDENT_OBSERVATION_HISTORY = 25
ACTION_SCALE = 0.25
ANGULAR_VELOCITY_SCALE = 0.25
JOINT_VELOCITY_SCALE = 0.05
OBSERVATION_CLIP = 100.0
ACTION_CLIP = 100.0

COMMAND_LIMITS = np.array(
    [
        [-0.75, 1.05],
        [-0.50, 0.50],
        [-1.20, 1.20],
        [0.45, 0.90],
    ],
    dtype=np.float32,
)

POLICY_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

DEFAULT_JOINT_POSITIONS = {
    "left_hip_pitch_joint": -0.20,
    "right_hip_pitch_joint": -0.20,
    "left_knee_joint": 0.42,
    "right_knee_joint": 0.42,
    "left_ankle_pitch_joint": -0.23,
    "right_ankle_pitch_joint": -0.23,
    "left_wrist_roll_joint": -np.pi / 2.0,
    "right_wrist_roll_joint": np.pi / 2.0,
}

PHASE1_POSE_COMMAND = np.array(
    [
        0.707,
        -0.707,
        0.0,
        0.0,
        0.707,
        0.707,
        0.0,
        0.0,
        0.2413,
        0.1517,
        0.0952,
        0.2413,
        -0.1516,
        0.0952,
    ],
    dtype=np.float32,
)

STUDENT_POSE_COMMAND = np.array(
    [
        0.70712793,
        -0.70708555,
        0.00008733,
        -0.00004816,
        0.70712793,
        0.70708561,
        0.00008731,
        0.00004810,
        0.2413,
        0.1517,
        0.0952,
        0.2413,
        -0.1516,
        0.0952,
    ],
    dtype=np.float32,
)

STUDENT_MASKED_COMMAND = np.array(
    [0.0, 0.0, 0.0, 0.78], dtype=np.float32
)

SCHEDULE = (
    (4.0, (0.0, 0.0, 0.0, 0.78)),
    (5.0, (0.4, 0.0, 0.0, 0.78)),
    (5.0, (0.8, 0.0, 0.0, 0.78)),
    (5.0, (-0.3, 0.0, 0.0, 0.78)),
    (5.0, (0.0, 0.3, 0.0, 0.78)),
    (5.0, (0.0, -0.3, 0.0, 0.78)),
    (5.0, (0.0, 0.0, 0.5, 0.78)),
    (5.0, (0.0, 0.0, -0.5, 0.78)),
    (5.0, (0.0, 0.0, 0.0, 0.55)),
    (5.0, (0.3, 0.3, 0.0, 0.78)),
    (4.0, (0.0, 0.0, 0.0, 0.78)),
)


def actuator_gains(joint_name: str) -> tuple[float, float]:
    """Return the nominal gains shared with the Isaac Sim training asset."""

    if "hip_roll" in joint_name or "knee" in joint_name:
        return 99.09842777666113, 6.3088018534966395
    if "hip_yaw" in joint_name or "hip_pitch" in joint_name:
        return 40.17923847137318, 2.5578897650279457
    if "waist_yaw" in joint_name:
        return 40.17923847137318, 2.5578897650279457
    if "waist_roll" in joint_name or "waist_pitch" in joint_name:
        return 28.50124619574858, 1.814445686584846
    if "ankle" in joint_name:
        return 28.50124619574858, 1.814445686584846
    if (
        "shoulder" in joint_name
        or "elbow" in joint_name
        or "wrist_roll" in joint_name
    ):
        return 14.25062309787429, 0.907222843292423
    if "wrist_pitch" in joint_name or "wrist_yaw" in joint_name:
        return 16.77832748089279, 1.06814150219
    raise KeyError(f"No actuator gains are defined for {joint_name!r}")


def rotate_inverse(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion)
    vector = np.asarray(vector, dtype=np.float64)
    scalar = quaternion[0]
    imaginary = quaternion[1:]
    return (
        vector * (2.0 * scalar * scalar - 1.0)
        - 2.0 * scalar * np.cross(imaginary, vector)
        + 2.0 * imaginary * np.dot(imaginary, vector)
    )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot access CUDA")
    return torch.device(requested)


class LocomotionSim2Sim:
    def __init__(self, args: argparse.Namespace) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
        self.model.opt.timestep = SIMULATION_DT
        self.data = mujoco.MjData(self.model)
        self.device = resolve_device(args.device)

        joint_ids = [
            joint_id
            for joint_id in range(self.model.njnt)
            if self.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        ]
        self.joint_names = [
            mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
            )
            for joint_id in joint_ids
        ]
        if len(self.joint_names) != NUM_ACTIONS:
            raise RuntimeError(
                f"Expected {NUM_ACTIONS} hinge joints, found {len(self.joint_names)}"
            )
        if set(self.joint_names) != set(POLICY_JOINT_ORDER):
            missing = sorted(set(POLICY_JOINT_ORDER) - set(self.joint_names))
            extra = sorted(set(self.joint_names) - set(POLICY_JOINT_ORDER))
            raise RuntimeError(f"Joint mismatch: missing={missing}, extra={extra}")

        self.qpos_addresses = np.array(
            [self.model.jnt_qposadr[joint_id] for joint_id in joint_ids]
        )
        self.dof_addresses = np.array(
            [self.model.jnt_dofadr[joint_id] for joint_id in joint_ids]
        )
        name_to_mujoco = {
            name: index for index, name in enumerate(self.joint_names)
        }
        self.policy_to_mujoco = np.array(
            [name_to_mujoco[name] for name in POLICY_JOINT_ORDER]
        )
        self.mujoco_to_policy = np.argsort(self.policy_to_mujoco)

        self.default_positions = np.array(
            [DEFAULT_JOINT_POSITIONS.get(name, 0.0) for name in self.joint_names]
        )
        gains = [actuator_gains(name) for name in self.joint_names]
        self.kp = np.array([gain[0] for gain in gains])
        self.kd = np.array([gain[1] for gain in gains])

        joint_to_actuator = {
            int(self.model.actuator_trnid[actuator_id, 0]): actuator_id
            for actuator_id in range(self.model.nu)
        }
        self.actuator_ids = np.array(
            [joint_to_actuator[joint_id] for joint_id in joint_ids]
        )
        self.base_qpos_address = int(
            self.model.joint("floating_base_joint").qposadr[0]
        )
        self.base_dof_address = int(
            self.model.joint("floating_base_joint").dofadr[0]
        )

        self.policy = torch.jit.load(
            str(args.policy.resolve()), map_location=self.device
        ).to(self.device).eval()
        self.policy_kind, self.observation_history_length = (
            self._detect_policy_contract()
        )
        self.command = np.asarray(args.command, dtype=np.float32)
        self._validate_command(self.command)
        if self.policy_kind == "phase3_student" and not np.array_equal(
            self.command, STUDENT_MASKED_COMMAND
        ):
            raise ValueError(
                "phase-3 no-object students were trained with the masked "
                "command [0, 0, 0, 0.78]; do not pass a different --command"
            )
        self.previous_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.target_positions = self.default_positions.copy()
        self.history: deque[np.ndarray] = deque(
            maxlen=self.observation_history_length
        )
        self.reset(args.initial_height)
        self._validate_policy()

    @torch.inference_mode()
    def _detect_policy_contract(self) -> tuple[str, int]:
        candidates = (
            ("phase1", PHASE1_OBSERVATION_HISTORY),
            ("phase3_student", STUDENT_OBSERVATION_HISTORY),
        )
        matches: list[tuple[str, int]] = []
        for kind, history_length in candidates:
            sample = torch.zeros(
                1,
                OBSERVATION_FRAME_SIZE * history_length,
                device=self.device,
            )
            try:
                output = self.policy(sample)
                if isinstance(output, (tuple, list)):
                    output = output[0]
                if tuple(output.shape) == (1, NUM_ACTIONS):
                    matches.append((kind, history_length))
            except (RuntimeError, ValueError):
                continue
        if len(matches) != 1:
            supported = (
                OBSERVATION_FRAME_SIZE * PHASE1_OBSERVATION_HISTORY,
                OBSERVATION_FRAME_SIZE * STUDENT_OBSERVATION_HISTORY,
            )
            raise RuntimeError(
                f"Policy must accept exactly one supported input width {supported}; "
                f"matched={matches}"
            )
        return matches[0]

    @staticmethod
    def _validate_command(command: np.ndarray) -> None:
        if command.shape != (4,):
            raise ValueError("command must contain vx, vy, wz, and height")
        if np.any(command < COMMAND_LIMITS[:, 0]) or np.any(
            command > COMMAND_LIMITS[:, 1]
        ):
            raise ValueError(
                f"command {command.tolist()} is outside the phase-1 ranges "
                f"{COMMAND_LIMITS.tolist()}"
            )

    def set_command(self, command: tuple[float, float, float, float]) -> None:
        value = np.asarray(command, dtype=np.float32)
        self._validate_command(value)
        self.command = value

    def reset(self, initial_height: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[
            self.base_qpos_address : self.base_qpos_address + 7
        ] = (0.0, 0.0, initial_height, 1.0, 0.0, 0.0, 0.0)
        self.data.qpos[self.qpos_addresses] = self.default_positions
        self.data.qvel.fill(0.0)
        self.data.ctrl.fill(0.0)
        mujoco.mj_forward(self.model, self.data)
        self.previous_action.fill(0.0)
        self.target_positions[:] = self.default_positions
        self.history.clear()

    def observation_frame(self) -> np.ndarray:
        joint_position = (
            self.data.qpos[self.qpos_addresses] - self.default_positions
        )[self.policy_to_mujoco]
        joint_velocity = self.data.qvel[self.dof_addresses][
            self.policy_to_mujoco
        ]
        base_quaternion = self.data.qpos[
            self.base_qpos_address + 3 : self.base_qpos_address + 7
        ]
        base_angular_velocity = self.data.qvel[
            self.base_dof_address + 3 : self.base_dof_address + 6
        ]
        projected_gravity = rotate_inverse(
            base_quaternion, np.array([0.0, 0.0, -1.0])
        )
        command = (
            STUDENT_MASKED_COMMAND
            if self.policy_kind == "phase3_student"
            else self.command
        )
        pose_command = (
            STUDENT_POSE_COMMAND
            if self.policy_kind == "phase3_student"
            else PHASE1_POSE_COMMAND
        )
        frame = np.concatenate(
            (
                command,
                pose_command,
                (base_angular_velocity * ANGULAR_VELOCITY_SCALE).astype(
                    np.float32
                ),
                projected_gravity.astype(np.float32),
                joint_position.astype(np.float32),
                (joint_velocity * JOINT_VELOCITY_SCALE).astype(np.float32),
                self.previous_action,
            )
        )
        if frame.shape != (OBSERVATION_FRAME_SIZE,):
            raise RuntimeError(f"Unexpected observation frame {frame.shape}")
        return frame

    def stacked_observation(self) -> torch.Tensor:
        frame = self.observation_frame()
        if not self.history:
            self.history.extend(
                frame.copy() for _ in range(self.observation_history_length)
            )
        else:
            self.history.append(frame)
        observation = np.clip(
            np.concatenate(tuple(self.history)),
            -OBSERVATION_CLIP,
            OBSERVATION_CLIP,
        )
        return torch.from_numpy(observation).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def infer_action(self) -> None:
        output = self.policy(self.stacked_observation())
        if isinstance(output, (tuple, list)):
            output = output[0]
        action = output.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if action.shape != (NUM_ACTIONS,) or not np.all(np.isfinite(action)):
            raise RuntimeError(f"Invalid policy action with shape {action.shape}")
        action = np.clip(action, -ACTION_CLIP, ACTION_CLIP)
        action_mujoco = action[self.mujoco_to_policy]
        self.target_positions = (
            self.default_positions + ACTION_SCALE * action_mujoco
        )
        self.previous_action = action

    def apply_pd(self) -> None:
        position = self.data.qpos[self.qpos_addresses]
        velocity = self.data.qvel[self.dof_addresses]
        torque = self.kp * (self.target_positions - position) - self.kd * velocity
        self.data.ctrl[self.actuator_ids] = torque

    def physics_step(self, step_index: int) -> None:
        if step_index % POLICY_DECIMATION == 0:
            self.infer_action()
        self.apply_pd()
        mujoco.mj_step(self.model, self.data)

    def _validate_policy(self) -> None:
        self.infer_action()
        self.previous_action.fill(0.0)
        self.target_positions[:] = self.default_positions
        self.history.clear()

    @property
    def base_height(self) -> float:
        return float(self.data.qpos[self.base_qpos_address + 2])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=DEFAULT_POLICY,
        help=(
            "Exported 1110-D phase-1 actor or 2775-D phase-3 student policy "
            "(default: deployment/mujoco/student.jit)."
        ),
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--command",
        type=float,
        nargs=4,
        metavar=("VX", "VY", "WZ", "HEIGHT"),
        default=(0.0, 0.0, 0.0, 0.78),
        help=(
            "Phase-1 command. Phase-3 no-object students require the masked "
            "default 0 0 0 0.78."
        ),
    )
    parser.add_argument("--initial-height", type=float, default=0.80)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    return parser.parse_args()


def run_fixed(
    simulation: LocomotionSim2Sim,
    duration: float,
    headless: bool,
) -> None:
    if headless and duration <= 0.0:
        raise ValueError("headless runs require a positive --duration")
    maximum_steps = (
        round(duration / SIMULATION_DT) if duration > 0.0 else None
    )
    start = time.perf_counter()
    step = 0

    if headless:
        while maximum_steps is None or step < maximum_steps:
            simulation.physics_step(step)
            step += 1
    else:
        from mujoco import viewer

        with viewer.launch_passive(simulation.model, simulation.data) as handle:
            while handle.is_running() and (
                maximum_steps is None or step < maximum_steps
            ):
                simulation.physics_step(step)
                step += 1
                if step % 5 == 0:
                    handle.sync()
                    delay = simulation.data.time - (time.perf_counter() - start)
                    if delay > 0.0:
                        time.sleep(delay)

    print(
        f"SIM2SIM_COMPLETE time={simulation.data.time:.3f}s "
        f"steps={step} base_height={simulation.base_height:.3f}m"
    )


def run_schedule(simulation: LocomotionSim2Sim) -> None:
    if simulation.policy_kind != "phase1":
        raise ValueError("--schedule is only available for phase-1 policies")
    step = 0
    for duration, command in SCHEDULE:
        simulation.set_command(command)
        segment_steps = round(duration / SIMULATION_DT)
        for _ in range(segment_steps):
            simulation.physics_step(step)
            step += 1
            if simulation.base_height < 0.35:
                raise RuntimeError(
                    f"Robot fell at t={simulation.data.time:.3f}s "
                    f"under command={command}"
                )
        print(
            f"SCHEDULE_SEGMENT_PASS command={command} "
            f"base_height={simulation.base_height:.3f}m"
        )
    print(f"SIM2SIM_SCHEDULE_PASS time={simulation.data.time:.3f}s")


def main() -> None:
    args = parse_args()
    simulation = LocomotionSim2Sim(args)
    print(
        f"model={args.model}\npolicy={args.policy}\n"
        f"physics={1.0 / SIMULATION_DT:.0f}Hz policy={1.0 / POLICY_DT:.0f}Hz "
        f"contract={simulation.policy_kind} "
        f"observation={OBSERVATION_FRAME_SIZE}x"
        f"{simulation.observation_history_length}"
    )
    if args.schedule:
        run_schedule(simulation)
    else:
        run_fixed(simulation, args.duration, args.headless)


if __name__ == "__main__":
    main()
