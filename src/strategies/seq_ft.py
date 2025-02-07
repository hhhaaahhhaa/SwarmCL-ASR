import os
import copy
import yaml
import random

from one import train_one_task, load_system
from src.strategies.base import IStrategy
from src.tasks.utils import TaskSequence


class SeqFTStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

    def _get_exp_root(self, tid: int):
        return f"{self.config['output_dir']['log_dir']}/tid={tid+1}"
    
    def _get_checkpoint_path(self, tid: int):
        return f"{self.config['output_dir']['ckpt_dir']}/tid={tid+1}.ckpt"

    def _get_ft_system(self, tid: int):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            checkpoint=f"{self._get_exp_root(tid)}/ckpt/best.ckpt",
            loader="lightning"
        )

    def _finetune(self, tid: int, task_name: str):
        # create full configuration
        task_config = {
            "system_name": self.config["strategy_config"]["system_name"],
            "task_name": task_name,
            "checkpoint": None if tid == 0 else self._get_checkpoint_path(tid-1),
            "config": copy.deepcopy(self.config["system_config"]),
        }
        exp_root = self._get_exp_root(tid)
        os.makedirs(exp_root, exist_ok=True)
        task_config["config"]["output_dir"] = {
            "exp_root": exp_root,
            "log_dir": f"{exp_root}/log",
            "result_dir": f"{exp_root}/result",
            "ckpt_dir": f"{exp_root}/ckpt"
        }
        with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(task_config, f, sort_keys=False)
        
        train_one_task(task_config, loader="torch")

    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert isinstance(data_obj, TaskSequence)
        for tid, task_name in enumerate(data_obj.task_names):
            self._finetune(tid, task_name)
            ft_system = self._get_ft_system(tid)
            ft_system.save(self._get_checkpoint_path(tid))


class ERStrategy(SeqFTStrategy):
    def __init__(self, config):
        super().__init__(config)
        self._buffer = []
        self._buffer_size = self.config["strategy_config"]["er"]["max_size"]

    def _update_buffer(self, samples: list):
        assert self._buffer_size > 0, "Please set a buffer size first!"
        self._buffer.extend(samples)
        if len(self._buffer) > self._buffer_size:
            self._buffer = random.sample(self._buffer, self._buffer_size)
        print(f"Current buffer size: {len(self._buffer)}.")

    def _finetune(self, tid: int, task_name: str):
        # create full configuration
        task_config = {
            "system_name": self.config["strategy_config"]["system_name"],
            "task_name": task_name,
            "checkpoint": None if tid == 0 else self._get_checkpoint_path(tid-1),
            "config": copy.deepcopy(self.config["system_config"]),
        }
        exp_root = self._get_exp_root(tid)
        os.makedirs(exp_root, exist_ok=True)
        task_config["config"]["output_dir"] = {
            "exp_root": exp_root,
            "log_dir": f"{exp_root}/log",
            "result_dir": f"{exp_root}/result",
            "ckpt_dir": f"{exp_root}/ckpt"
        }
        with open(f"{exp_root}/config.yaml", "w", encoding="utf-8") as f:
            yaml.dump(task_config, f, sort_keys=False)
        
        def hook(system):
            system._buffer = self._buffer
        train_one_task(task_config, loader="torch", system_hook=hook)

    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert isinstance(data_obj, TaskSequence)
        for tid, task_name in enumerate(data_obj.task_names):
            self._finetune(tid, task_name)
            ft_system = self._get_ft_system(tid)
            ft_system.save(self._get_checkpoint_path(tid))

            # update buffer
            ds = data_obj.tasks[tid].val_dataset()
            self._update_buffer([ds[i] for i in range(self.config["strategy_config"]["er"]["n_per_task"])])
