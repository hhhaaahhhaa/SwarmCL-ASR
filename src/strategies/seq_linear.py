import os
import copy
import yaml

from one import train_one_task, load_system
from src.strategies.base import IStrategy
from .common.particle import linear_combination, system2particle, particle2system


class SeqLinearStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

    def _get_exp_root(self, tid: int):
        return f"{self.config['output_dir']['log_dir']}/tid={tid+1}"
    
    def _get_checkpoint_path(self, tid: int):
        return f"{self.config['output_dir']['ckpt_dir']}/tid={tid+1}.ckpt"
    
    def _get_initial_system(self):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"])
        )

    def _get_ft_system(self, tid: int):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            checkpoint=f"{self._get_exp_root(tid)}/ckpt/best.ckpt",
            loader="lightning"
        )

    def _get_merged_system(self, tid: int):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"]),
            checkpoint=self._get_checkpoint_path(tid),
            loader="torch"
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
        assert task_name in ["cv-seq"]
        for tid, task_name in enumerate(data_obj.task_names):
            self._finetune(tid, task_name)

            # merge
            prev_system = self._get_merged_system(tid-1) if tid > 0 else self._get_initial_system()
            prev_particle = system2particle(prev_system)
            del prev_system
            ft_system = self._get_ft_system(tid)
            ft_particle = system2particle(ft_system)
            del ft_system

            r = self.config["strategy_config"]["lambda"]
            merged_particle = linear_combination([1 - r, r], [prev_particle, ft_particle])
            merged_system = particle2system(
                merged_particle,
                ref_system=self._get_initial_system()
            )
            merged_system.save(self._get_checkpoint_path(tid))
