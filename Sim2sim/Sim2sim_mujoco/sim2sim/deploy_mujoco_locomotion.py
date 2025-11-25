import os
import argparse


import numpy as np
import torch
import faulthandler
import matplotlib.pyplot as plt

import time
from collections import deque
import transforms3d as t3d

import mujoco
import mujoco.viewer
from pynput import keyboard

# Control instructions
control_guide = """
🔹 Robot Keyboard Control Guide 🔹
---------------------------------
[8]  Move Forward   [2]  Move Backward
[4]  Move Left      [6]  Move Right
[7]  Turn Left      [9]  Turn Right
[5]  Stop
[ESC] Exit Program
---------------------------------
"""

print(control_guide)

XYYAW = {"x": 0.0, "y": 0.0, "yaw": 0.0}

def on_press(key):
    try:
        if key.char =="8":  # 按 8 前进
            XYYAW["x"] += 0.1  # 控制机器人，例如移动
            print("cmd:", XYYAW)
        elif key.char =="2":  # 按 2 后退
            XYYAW["x"] -= 0.1
            print("cmd:", XYYAW)
        elif key.char =="4":  # 按 4 左平移
            XYYAW["y"] += 0.1
            print("cmd:", XYYAW)
        elif key.char =="6":  # 按 6 右平移
            XYYAW["y"] -= 0.1
            print("cmd:", XYYAW)
        elif key.char =="7":  # 按 7 左转
            XYYAW["yaw"] += 0.1
            print("cmd:", XYYAW)
        elif key.char =="9":  # 按 9 右转
            XYYAW["yaw"] -= 0.1
            print("cmd:", XYYAW)
        elif key.char =="5":  # 按 5 停止
            XYYAW["x"] = 0.0
            XYYAW["y"] = 0.0
            XYYAW["yaw"] = 0.0
    except AttributeError:
        pass  # 忽略特殊键

# 启动监听线程
listener = keyboard.Listener(on_press=on_press)
listener.start()

HW_DOF = 29

WALK_STRAIGHT = False
LOG_DATA = False
USE_GRIPPPER = False
NO_MOTOR = False

HUMANOID_XML = "./sim2sim/assets/robots/g1/scene_29dof.xml"
DEBUG = True
SIM = True

def add_visual_capsule(scene, point1, point2, radius, rgba):
    """Adds one capsule to an mjvScene."""
    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1  # increment ngeom
    # initialise a new capsule, add it to the scene using mjv_makeConnector
    mujoco.mjv_initGeom(scene.geoms[scene.ngeom-1],
                        mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3),
                        np.zeros(3), np.zeros(9), rgba.astype(np.float32))
    mujoco.mjv_makeConnector(scene.geoms[scene.ngeom-1],
                            mujoco.mjtGeom.mjGEOM_CAPSULE, radius,
                            point1[0], point1[1], point1[2],
                            point2[0], point2[1], point2[2])
    
def quat_rotate_inverse(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_w = q[..., 0]
    q_vec = q[..., 1:]
    a = v * (2.0 * q_w**2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    # for two-dimensional tensors, bmm is faster than einsum
    if q_vec.dim() == 2:
        c = q_vec * torch.bmm(q_vec.view(q.shape[0], 1, 3), v.view(q.shape[0], 3, 1)).squeeze(-1) * 2.0
    else:
        c = q_vec * torch.einsum("...i,...i->...", q_vec, v).unsqueeze(-1) * 2.0
    return a - b + c

class G1():
    def __init__(self,task='stand'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.task = task


        self.num_envs = 1 
        # self.num_observations = 104#95
        self.num_observations = 111
        self.num_actions = 29
        self.num_privileged_obs = None
        self.obs_context_len=10
        
        self.scale_lin_vel = 2.0
        self.scale_ang_vel = 0.25
        self.scale_orn = 1.0
        self.scale_dof_pos = 1.0
        self.scale_dof_vel = 0.05
        self.scale_action = 0.25
        
        # prepare gait commands
        self.cycle_time = 0.64
        self.gait_indices = torch.zeros(self.num_envs, dtype=torch.float, device=self.device, requires_grad=False)
        
        # prepare action deployment joint positions offsets and PD gains
        self.p_gains = np.array([
            80., 80., 80., 160., 20., 20., 80., 80., 80., 160., 20., 20., 200., 200., 200.,
            40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40., 40.
        ])
        self.d_gains = np.array([
            2., 2., 2., 4., 0.5, 0.5, 2., 2., 2., 4., 0.5, 0.5, 5., 5., 5.,
            1., 1., 1., 1., 0.5, 0.5, 0.5, 1., 1., 1., 1., 0.5, 0.5, 0.5
        ])

        self.mujoco2policy_action = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        self.policy2mujoco_action = [ 0,  3,  6,  9, 13, 17,  1,  4,  7, 10, 14, 18,  2,  5,  8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]

        self.joint_limit_lo = [-2.5307, -0.5236, -2.7576, -0.087267, -np.inf, -np.inf, -2.5307,-2.9671,-2.7576,-0.087267,-np.inf,-np.inf,-2.618,-0.52,-0.52,-3.0892,-1.5882,-2.618,-1.0472, -1.972222054,-1.614429558,-1.614429558,-3.0892,-2.2515,-2.618,-1.0472,-1.972222054,-1.614429558,-1.614429558]
        self.joint_limit_hi = [2.8798, 2.9671, 2.7576, 2.8798, np.inf, np.inf, 2.8798, 0.5236, 2.7576, 2.8798, np.inf, np.inf, 2.618, 0.52, 0.52,2.6704,2.2515,2.618,2.0944,1.972222054,1.614429558,1.614429558,2.6704,1.5882,2.618,2.0944, 1.972222054,1.614429558,1.614429558]
        self.soft_dof_pos_limit = 1.0
        for i in range(len(self.joint_limit_lo)):
            # soft limits
            if i != 5 and i != 11 and i !=4 and i != 10:
                m = (self.joint_limit_lo[i] + self.joint_limit_hi[i]) / 2
                r = self.joint_limit_hi[i] - self.joint_limit_lo[i]
                self.joint_limit_lo[i] = m - 0.5 * r * self.soft_dof_pos_limit
                self.joint_limit_hi[i] = m + 0.5 * r * self.soft_dof_pos_limit
        
        self.default_dof_pos_np = np.zeros(29)

        self.default_dof_pos_np[:29] = np.array([
                                            -0.2, #left hip pitch
                                            0.0, #left hip roll
                                            0.0, #left hip pitch
                                            0.42, #left knee
                                            -0.23, #left ankle pitch 
                                            0, #left ankle roll 
                                            -0.2, #right hip pitch
                                            0.0, #right hip roll
                                            0.0, #right hip pitch
                                            0.42, #right knee
                                            -0.23, #right ankle pitch 
                                            0, #right ankle roll 
                                            0, #waist
                                            0, #waist
                                            0, #waist
                                            0.,
                                            0.4,
                                            0.,
                                            0.,
                                            -1.57,
                                            0.,
                                            0.,
                                            0.,
                                            -0.4,
                                            0.,
                                            0.,
                                            1.57,
                                            0.,
                                            0.,
                                            ])
        
        default_dof_pos = torch.tensor(self.default_dof_pos_np, dtype=torch.float, device=self.device, requires_grad=False)
        self.default_dof_pos = default_dof_pos.unsqueeze(0)

        # prepare osbervations buffer
        self.obs_buf = torch.zeros(1, self.num_observations*self.obs_context_len, dtype=torch.float, device=self.device, requires_grad=False)
        self.obs_history = deque(maxlen=self.obs_context_len)
        for _ in range(self.obs_context_len):
            self.obs_history.append(torch.zeros(
                1, self.num_observations, dtype=torch.float, device=self.device))
    
    def init_mujoco_viewer(self):

        self.mj_model = mujoco.MjModel.from_xml_path(HUMANOID_XML)

        self.mj_model.dof_damping[:] = 1.0
        self.mj_model.dof_armature[:] = 0.01

        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = 0.001
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)


        for _ in range(28):
            add_visual_capsule(self.viewer.user_scn, np.zeros(3), np.array([0.001, 0, 0]), 0.05, np.array([0, 1, 0, 1]))
        self.viewer.user_scn.geoms[27].pos = [0,0,0]

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

class DeployNode():

    class WirelessButtons:
        R1 =            0b00000001 # 1
        L1 =            0b00000010 # 2
        start =         0b00000100 # 4
        select =        0b00001000 # 8
        R2 =            0b00010000 # 16
        L2 =            0b00100000 # 32
        F1 =            0b01000000 # 64
        F2 =            0b10000000 # 128
        A =             0b100000000 # 256
        B =             0b1000000000 # 512
        X =             0b10000000000 # 1024
        Y =             0b100000000000 # 2048
        up =            0b1000000000000 # 4096
        right =         0b10000000000000 # 8192
        down =          0b100000000000000 # 16384
        left =          0b1000000000000000 # 32768

    def __init__(self):
        
        # init subcribers & publishers
        # self.joy_stick_sub = self.create_subscription(WirelessController, "wirelesscontroller", self._joy_stick_callback, 1)
        # self.joy_stick_sub  # prevent unused variable warning
        
        self.joint_pos = np.zeros(HW_DOF)
        self.joint_vel = np.zeros(HW_DOF)

        self.motor_pub_freq = 50
        self.dt = 1/self.motor_pub_freq


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.init_policy()
        self.prev_action = np.zeros(self.env.num_actions)
        self.start_policy = True
        if DEBUG:
            self.env.init_mujoco_viewer()
            self.env.mj_data.qpos[7:29+7] = self.angles
            self.env.mj_data.qpos[:3] = [0, 0, 0.78]
            mujoco.mj_forward(self.env.mj_model, self.env.mj_data)

            tau = pd_control(self.angles, 
                            self.env.mj_data.qpos[7:29+7], 
                            self.env.p_gains, 
                            np.zeros(self.env.num_actions), 
                            self.env.mj_data.qvel[6:29+6], 
                            self.env.d_gains)
            self.env.mj_data.ctrl[:] = tau
                        # mj_step can be replaced with code that also evaluates
                        # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            
            self.env.viewer.sync()
        self.stand_up = False
        self.stand_up = True

        self.current_step_count = 0

        # commands 
        self.lin_vel_deadband = 0.1
        self.ang_vel_deadband = 0.1
        self.move_by_wireless_remote = True
        self.cmd_px_range = [0.1, 2.5]
        self.cmd_nx_range = [0.1, 2]
        self.cmd_py_range = [0.1, 0.5]
        self.cmd_ny_range = [0.1, 0.5]
        self.cmd_pyaw_range = [0.2, 1.0]
        self.cmd_nyaw_range = [0.2, 1.0]

        # start
        self.init_buffer = 0
        self.foot_contact_buffer = []
        self.time_hist = []
        self.obs_time_hist = []
        self.angle_hist = []
        self.action_hist = []
        self.dof_pos_hist = []
        self.dof_vel_hist = []
        self.imu_hist = []
        self.ang_vel_hist = []
        self.foot_contact_hist = []
        self.tau_hist = []
        self.obs_hist = []

        self.count = 0


        # cmd and observation
        self.xyyaw_command = np.array([0., 0., 0.], dtype= np.float32)
        self.height_command = np.array([0.8], dtype= np.float32)
        self.commands_scale = np.array([self.env.scale_lin_vel, self.env.scale_lin_vel, self.env.scale_ang_vel])

        self.mujoco2policy_action = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]
        self.policy2mujoco_action = [ 0,  3,  6,  9, 13, 17,  1,  4,  7, 10, 14, 18,  2,  5,  8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]

        self.episode_length_buf = torch.zeros(1, device=self.device, dtype=torch.long)
        self.phase = torch.zeros(1, device=self.device, dtype=torch.float)

        self.Emergency_stop = False
        self.stop = False

        time.sleep(1)
    
        
    ##############################
    # subscriber callbacks
    ##############################


    def lowlevel_state_mujoco(self):
        if DEBUG and self.start_policy and SIM:
            # imu data
            quat = self.env.mj_data.qpos[3:7]
            self.obs_ang_vel = np.array(self.env.mj_data.qvel[3:6])*self.env.scale_ang_vel

            self.projected_gravity = quat_rotate_inverse(torch.tensor(quat, device=self.device).unsqueeze(0), torch.tensor(np.array([0, 0, -1.]), device=self.device).unsqueeze(0)).squeeze(0).cpu().numpy()

            euler = t3d.euler.quat2euler(quat)
            self.roll, self.pitch, self.yaw = euler[0], euler[1], euler[2]
            self.obs_imu = np.array([self.roll, self.pitch, self.yaw])*self.env.scale_orn

            # motor data
            self.joint_pos = self.env.mj_data.qpos[7:29+7]
            self.obs_joint_pos = (np.array(self.joint_pos) - self.env.default_dof_pos_np) * self.env.scale_dof_pos
            self.joint_vel = self.env.mj_data.qvel[6:29+6]
            self.obs_joint_vel = np.array(self.joint_vel) * self.env.scale_dof_vel

            
            self.xyyaw_command=np.array([XYYAW["x"], XYYAW["y"], XYYAW["yaw"]])



            
    ##############################
    # motor commands
    ##############################

    ##############################
    # deploy policy
    ##############################
    def init_policy(self):
        faulthandler.enable()

        # prepare environment
        self.env = G1(task='self.task')

        # load policy
        self.policy = torch.jit.load("./models/locomotion_actor_height.pt", map_location=self.env.device)
        self.policy.to(self.env.device)
        # init p_gains, d_gains, torque_limits
        self.angles = self.env.default_dof_pos_np
    
    def get_walking_cmd_mask(self):
        walking_mask0 = np.abs(self.xyyaw_command[0]) > 0.1
        walking_mask1 = np.abs(self.xyyaw_command[1]) > 0.1
        walking_mask2 = np.abs(self.xyyaw_command[2]) > 0.2
        walking_mask = walking_mask0 | walking_mask1 | walking_mask2

        walking_mask = walking_mask | (self.env.gait_indices.cpu() >= self.dt / self.env.cycle_time).numpy()[0]
        walking_mask |= np.logical_or(np.abs(self.obs_imu[1])>0.1, np.abs(self.obs_imu[0])>0.05)
        return walking_mask
    
    def  _get_phase(self):
        phase = self.env.gait_indices
        return phase
    
    def step_contact_targets(self):
        cycle_time = self.env.cycle_time
        standing_mask = ~self.get_walking_cmd_mask()
        self.env.gait_indices = torch.remainder(self.env.gait_indices + self.dt / cycle_time, 1.0)
        if standing_mask:
            self.env.gait_indices[:] = 0

    def compute_one_frame_observations(self):

        center_left_xyz = np.array([0.2413, 0.1517, 0.0952])
        center_right_xyz = np.array([0.2413, -0.1517, 0.0952])
        center_left_quat = np.array([0.707, -0.707, 0, 0])
        center_right_quat = np.array([0.707, 0.707, 0, 0])
        
        obs_joint_pos_policy_order = self.obs_joint_pos[self.mujoco2policy_action]
        obs_joint_vel_policy_order = self.obs_joint_vel[self.mujoco2policy_action]

        obs_buf = torch.tensor(np.concatenate((
                            self.xyyaw_command, # dim 3,
                            self.height_command, # dim 1
                            center_left_quat, # dim 4
                            center_right_quat, # dim 4
                            center_left_xyz, # dim 3
                            center_right_xyz, # dim 3
                            self.obs_ang_vel,  # dim 3
                            self.projected_gravity, # dim 3
                            obs_joint_pos_policy_order, # dim 29
                            obs_joint_vel_policy_order, # dim 29
                            self.prev_action, # dim 29
                            ), axis=-1), dtype=torch.float, device=self.device).unsqueeze(0)
        
        self.current_step_count += 1
        # add perceptive inputs if not blind

        obs_now = obs_buf.clone()

        if self.episode_length_buf == 0:
            self.env.obs_history.clear()
            for _ in range(self.env.obs_history.maxlen):
                self.env.obs_history.append(obs_now.clone())
            obs_buf_all = torch.cat([self.env.obs_history[i]
                                   for i in range(self.env.obs_history.maxlen)], dim=-1)  # N,T,K
            self.env.obs_buf = obs_buf_all
            pass

        self.env.obs_history.append(obs_now)

        obs_buf_all = torch.cat([self.env.obs_history[i]
                                   for i in range(self.env.obs_history.maxlen)], dim=-1)  # N,T,K
        
        self.env.obs_buf = obs_buf_all
        pass

    def check_observation(self, actor_obs, mujoco_obs):
        print(np.all((actor_obs.squeeze(0) - mujoco_obs) == 0))
        print(f"check_obs{actor_obs.squeeze(0) - mujoco_obs}")
        return

    def check_action(self, actor_action, mujoco_action):
        print(np.all((actor_action - mujoco_action) == 0))
        print(f"check_action{actor_action - mujoco_action}")
        return

    @torch.no_grad()
    def main_loop(self):
        # keep stand up pose first
        _percent_1 = 0
        _duration_1 = 500
        firstRun = True
        init_success = False
        while self.stand_up and not self.start_policy:
        # while True:
            if firstRun:
                firstRun = False
                start_pos = self.joint_pos
            else:
                if _percent_1 < 1:
                    self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(self.env.default_dof_pos_np))
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                if _percent_1 == 1 and not init_success:
                    init_success = True
                    print("---Initialized---")
                if not NO_MOTOR:
                    self.motor_pub.publish(self.cmd_msg)

        cnt = 0
        fps_ckt = time.monotonic()

        
        while True:
            loop_start_time = time.monotonic()
            
            if self.Emergency_stop:
                breakpoint()
            if self.stop:
                _percent_1 = 0
                _duration_1 = 1000
                start_pos = self.joint_pos
                while _percent_1 < 1:
                    self.set_motor_position(q=(1 - _percent_1) * np.array(start_pos) + _percent_1 * np.array(self.env.default_dof_pos_np))
                    _percent_1 += 1 / _duration_1
                    _percent_1 = min(1, _percent_1)
                break

            if self.start_policy:
                if LOG_DATA:
                    self.dof_pos_hist.append(self.obs_joint_pos_)
                    self.dof_vel_hist.append(self.obs_joint_vel_)
                    self.imu_hist.append(self.obs_imu)
                    self.ang_vel_hist.append(self.obs_ang_vel)
                    self.tau_hist.append(self.joint_tau)
                    self.obs_hist.append(self.obs_buf_np)
                
                if DEBUG and SIM:
                    self.lowlevel_state_mujoco()
                self.step_contact_targets()
                self.compute_one_frame_observations()
                self.episode_length_buf += 1

                raw_actions = self.policy(self.env.obs_buf.detach().reshape(1, -1))
                self.prev_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)
                whole_body_action = raw_actions.clone().detach().cpu().numpy().squeeze(0)[self.policy2mujoco_action]

                angles = whole_body_action * self.env.scale_action + self.env.default_dof_pos_np
                self.angles = np.clip(angles, self.env.joint_limit_lo, self.env.joint_limit_hi)
                inference_time=time.monotonic()-loop_start_time
                # while 0.009-time.monotonic()+loop_start_time > 0:
                #     pass
                if not NO_MOTOR and not DEBUG:
                    self.motor_pub.publish(self.cmd_msg)
                    pass
                else:
                    if not SIM:
                        self.env.mj_data.qpos[7:29+7] = self.angles
                        mujoco.mj_forward(self.env.mj_model, self.env.mj_data)
                        self.env.viewer.sync()
                    else:
                        for i in range(20):
                            self.env.viewer.sync()
                            tau = pd_control(self.angles, 
                                                self.env.mj_data.qpos[7:29+7], 
                                                self.env.p_gains, 
                                                np.zeros(self.env.num_actions), 
                                                self.env.mj_data.qvel[6:29+6], 
                                                self.env.d_gains)
                            self.env.mj_data.ctrl[:] = tau
                            # mj_step can be replaced with code that also evaluates
                            # a policy and applies a control signal before stepping the physics.
                            mujoco.mj_step(self.env.mj_model, self.env.mj_data)
            while 0.02-time.monotonic()+loop_start_time>0:  #0.012473  0.019963
                pass
            cnt+=1
            if cnt == 500:
                dt = (time.monotonic()-fps_ckt)/cnt
                cnt = 0
                fps_ckt = time.monotonic()
                print(f"FPS: {1/dt}")


if __name__ == "__main__":
    dp_node = DeployNode()

    dp_node.main_loop()