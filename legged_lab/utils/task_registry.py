from typing import Any

from rsl_rl.env import VecEnv


class TaskRegistry:
    def __init__(self):
        self.task_classes = {}
        self.env_cfgs = {}
        self.train_cfgs = {}

    def register(self, name: str, task_class: type[VecEnv], env_cfg: Any, train_cfg: Any) -> None:
        self.task_classes[name] = task_class
        self.env_cfgs[name] = env_cfg
        self.train_cfgs[name] = train_cfg

    def get_task_class(self, name: str) -> type[VecEnv]:
        return self.task_classes[name]

    def get_cfgs(self, name: str) -> tuple[Any, Any]:
        train_cfg = self.train_cfgs[name]
        env_cfg = self.env_cfgs[name]
        return env_cfg, train_cfg


# make global task registry
task_registry = TaskRegistry()
