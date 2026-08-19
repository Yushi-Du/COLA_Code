"""Train a phase-2 teacher or phase-3 student on static scene populations."""

import argparse

from isaaclab.app import AppLauncher
import legged_lab.utils.cli_args as cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Train static COLA populations.")
parser.add_argument(
    "--phase",
    type=int,
    choices=(2, 3),
    required=True,
    help="Train the phase-2 teacher or phase-3 student.",
)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--no_object_rank_count",
    type=int,
    default=2,
    help="Number of highest global ranks assigned to no-object scenes.",
)
parser.add_argument(
    "--right_fixed_rank_count",
    type=int,
    default=3,
    help="Number of middle ranks assigned to right-weld scenes.",
)

cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from legged_lab.utils.task_registry import task_registry
from legged_lab.utils.app import run_with_simulation_app
from legged_lab.utils.static_population import (
    assign_three_static_populations,
)
from rsl_rl.runners import OnPolicyRunnerWholePipeResi
from isaaclab.utils.io import dump_yaml

from legged_lab.envs import *  # noqa: F401, F403
from legged_lab.utils.cli_args import update_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path

import os
import re

import torch


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


TASKS = {
    2: {
        "left": "cola_phase_2_teacher_left_fixed_bar",
        "right": "cola_phase_2_teacher_right_fixed_bar",
        "no_object": "cola_phase_2_teacher_no_object",
    },
    3: {
        "left": "cola_phase_3_student_left_fixed_bar",
        "right": "cola_phase_3_student_right_fixed_bar",
        "no_object": "cola_phase_3_student_no_object",
    },
}


def _safe_run_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not component:
        raise ValueError("distributed static-mix run identifier is empty")
    return component


def _validate_runtime_contract(
    runner: OnPolicyRunnerWholePipeResi,
    expected_no_object_ranks: int,
    expected_right_fixed_ranks: int,
) -> None:
    """Fail all ranks if shapes or topology counts differ from the contract."""

    local = torch.tensor(
        [
            runner.num_obs,
            runner.num_privileged_obs,
            runner.env.num_actions,
            int(runner.env.cola_topology_id),
        ],
        device=runner.device,
        dtype=torch.int64,
    )
    gathered = [torch.empty_like(local) for _ in range(runner.gpu_world_size)]
    torch.distributed.all_gather(gathered, local)
    contract = torch.stack(gathered).cpu()

    if not torch.all(contract[:, :3] == contract[0, :3]):
        raise RuntimeError(
            "static populations expose different policy shapes: "
            f"{contract.tolist()}"
        )
    no_object_count = int((contract[:, 3] == 1).sum().item())
    fixed_bar_count = int((contract[:, 3] == 0).sum().item())
    right_fixed_count = int((contract[:, 3] == 2).sum().item())
    if no_object_count != expected_no_object_ranks:
        raise RuntimeError(
            f"expected {expected_no_object_ranks} no-object ranks, "
            f"observed {no_object_count}: {contract.tolist()}"
        )
    if right_fixed_count != expected_right_fixed_ranks:
        raise RuntimeError(
            f"expected {expected_right_fixed_ranks} right-fixed ranks, "
            f"observed {right_fixed_count}: {contract.tolist()}"
        )
    if fixed_bar_count + right_fixed_count + no_object_count != runner.gpu_world_size:
        raise RuntimeError(f"unknown topology id in contract: {contract.tolist()}")
    if runner.gpu_global_rank == 0:
        print(
            "[STATIC-MIX] CONTRACT_PASS "
            f"left_fixed_bar_ranks={fixed_bar_count} "
            f"right_fixed_bar_ranks={right_fixed_count} "
            f"no_object_ranks={no_object_count} "
            f"physical_no_object_share={no_object_count / runner.gpu_world_size:.6f} "
            f"obs={int(contract[0, 0])} "
            f"privileged_obs={int(contract[0, 1])} "
            f"actions={int(contract[0, 2])}",
            flush=True,
        )


def _validate_synchronized_policy(runner: OnPolicyRunnerWholePipeResi) -> None:
    """Verify that heterogeneous rollouts still produced one shared policy."""

    maximum_difference = torch.zeros(1, device=runner.device)
    for parameter in runner.alg.policy.parameters():
        rank_zero_parameter = parameter.detach().clone()
        torch.distributed.broadcast(rank_zero_parameter, src=0)
        difference = torch.max(torch.abs(parameter.detach() - rank_zero_parameter))
        maximum_difference = torch.maximum(maximum_difference, difference.reshape(1))
    torch.distributed.all_reduce(
        maximum_difference, op=torch.distributed.ReduceOp.MAX
    )
    if not torch.isfinite(maximum_difference).all() or maximum_difference.item() > 1.0e-7:
        raise RuntimeError(
            "static-mix policies diverged across ranks: "
            f"maximum parameter difference={maximum_difference.item():.9e}"
        )
    if runner.gpu_global_rank == 0:
        print(
            "[STATIC-MIX] POLICY_SYNC_PASS "
            f"max_parameter_difference={maximum_difference.item():.9e}",
            flush=True,
        )


def train():
    if not args_cli.distributed:
        raise ValueError(
            "static mixed-topology training requires --distributed; use the "
            "individual registered task for a one-GPU smoke test"
        )

    world_size = int(os.environ["WORLD_SIZE"])
    global_rank = int(os.environ["RANK"])
    assignment = assign_three_static_populations(
        global_rank,
        world_size,
        no_object_rank_count=args_cli.no_object_rank_count,
        right_fixed_rank_count=args_cli.right_fixed_rank_count,
    )

    tasks = TASKS[args_cli.phase]
    fixed_bar_env_cfg, agent_cfg = task_registry.get_cfgs(tasks["left"])
    right_fixed_env_cfg, _ = task_registry.get_cfgs(tasks["right"])
    no_object_env_cfg, _ = task_registry.get_cfgs(tasks["no_object"])
    if assignment.is_no_object:
        env_cfg = no_object_env_cfg
        env_class = task_registry.get_task_class(tasks["no_object"])
        selected_task = tasks["no_object"]
    elif assignment.is_right_fixed:
        env_cfg = right_fixed_env_cfg
        env_class = task_registry.get_task_class(tasks["right"])
        selected_task = tasks["right"]
    else:
        env_cfg = fixed_bar_env_cfg
        env_class = task_registry.get_task_class(tasks["left"])
        selected_task = tasks["left"]

    if args_cli.num_envs is not None:
        fixed_bar_env_cfg.scene.num_envs = args_cli.num_envs
        no_object_env_cfg.scene.num_envs = args_cli.num_envs
        right_fixed_env_cfg.scene.num_envs = args_cli.num_envs

    agent_cfg = update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.device = f"cuda:{app_launcher.local_rank}"
    env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    agent_cfg.device = f"cuda:{app_launcher.local_rank}"
    rank_seed = agent_cfg.seed + global_rank
    env_cfg.scene.seed = rank_seed
    agent_cfg.seed = rank_seed

    print(
        "[STATIC-MIX] "
        f"rank={global_rank}/{world_size} local_rank={app_launcher.local_rank} "
        f"topology={assignment.topology_name} task={selected_task} "
        f"num_envs={env_cfg.scene.num_envs} seed={rank_seed}",
        flush=True,
    )
    env = env_class(env_cfg, args_cli.headless)
    env.cola_topology_id = assignment.topology_id
    env.cola_topology_name = assignment.topology_name

    log_root_path = os.path.abspath(
        os.path.join("logs", agent_cfg.experiment_name)
    )
    shared_run_id = os.environ.get("COLA_STATIC_MIX_RUN_ID") or agent_cfg.run_name
    if not shared_run_id:
        raise ValueError(
            "distributed static-mix training requires --run_name or "
            "COLA_STATIC_MIX_RUN_ID so every rank uses one log directory"
        )
    log_name = "static_mix_" + _safe_run_component(shared_run_id)
    log_dir = os.path.join(log_root_path, log_name)

    runner = OnPolicyRunnerWholePipeResi(
        env,
        agent_cfg.to_dict(),
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    _validate_runtime_contract(
        runner,
        args_cli.no_object_rank_count,
        args_cli.right_fixed_rank_count,
    )

    if agent_cfg.resume:
        if "/" in agent_cfg.load_run:
            resume_path = get_checkpoint_path(
                os.path.dirname(agent_cfg.load_run),
                os.path.basename(agent_cfg.load_run),
                agent_cfg.load_checkpoint,
            )
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                agent_cfg.load_run,
                agent_cfg.load_checkpoint,
            )
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(
            resume_path,
            load_optimizer=not args_cli.warm_start,
            reset_iteration=args_cli.warm_start,
        )

    if global_rank == 0:
        os.makedirs(os.path.join(log_dir, "params"), exist_ok=True)
        dump_yaml(os.path.join(log_dir, "params", "env_fixed_bar.yaml"), fixed_bar_env_cfg)
        dump_yaml(os.path.join(log_dir, "params", "env_no_object.yaml"), no_object_env_cfg)
        dump_yaml(
            os.path.join(log_dir, "params", "env_right_fixed.yaml"),
            right_fixed_env_cfg,
        )
        dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    runner.learn(
        num_learning_iterations=agent_cfg.max_iterations,
        init_at_random_ep_len=True,
    )
    _validate_synchronized_policy(runner)


if __name__ == "__main__":
    run_with_simulation_app(simulation_app, train)
