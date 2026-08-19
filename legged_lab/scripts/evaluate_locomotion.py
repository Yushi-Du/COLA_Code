import os

import argparse

from isaaclab.app import AppLauncher
import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate a phase-1 COLA checkpoint.")
parser.add_argument(
    "--task",
    type=str,
    default="cola_phase_1_locomotion",
    help="Registered locomotion task.",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from isaaclab_rl.rsl_rl import export_policy_as_jit, export_policy_as_onnx
from legged_lab.utils.task_registry import task_registry
from legged_lab.utils.app import run_with_simulation_app
from rsl_rl.runners import OnPolicyRunnerEnd2end

from legged_lab.envs import *  # noqa:F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path


def evaluate():
    runner: OnPolicyRunnerEnd2end
    env_class_name = args_cli.task
    env_cfg, agent_cfg = task_registry.get_cfgs(env_class_name)

    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.events.push_robot = None
    env_cfg.scene.max_episode_length_s = 40.0
    env_cfg.scene.num_envs = 1
    env_cfg.scene.env_spacing = 2.5
    env_cfg.scene.height_scanner.drift_range = (0.0, 0.0)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.seed = agent_cfg.seed

    env_class = task_registry.get_task_class(env_class_name)
    env = env_class(env_cfg, args_cli.headless)

    log_root_path = os.path.join("logs", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if "/" in agent_cfg.load_run:
        resume_path = get_checkpoint_path(
            os.path.dirname(agent_cfg.load_run),
            os.path.basename(agent_cfg.load_run),
            agent_cfg.load_checkpoint,
        )
    else:
        resume_path = get_checkpoint_path(
            log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
        )
    log_dir = os.path.dirname(resume_path)

    runner = OnPolicyRunnerEnd2end(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)

    policy = runner.get_inference_policy(device=env.device)

    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(runner.alg.policy, runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(runner.alg.policy, normalizer=runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    if not args_cli.headless:
        from legged_lab.utils.keyboard import Keyboard
        keyboard = Keyboard(env)  # noqa:F841

    obs, _ = env.get_observations()

    while simulation_app.is_running():

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)


if __name__ == "__main__":
    run_with_simulation_app(simulation_app, evaluate)
