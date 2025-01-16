import numpy as np
import torch
from tqdm import tqdm
import copy
import yaml
import json
from functools import partial

from one import load_system
from src.systems.load import get_system_cls
from src.strategies.base import IStrategy
from src.utils.tool import wer
from .common.greedysoup import GreedySoupExecutor
from .common.swarm import SwarmExecutor
from .common import merging
from .common.particle import ModelParticle, linear_combination, system2particle, particle2system
from .common.utils import load_exp0_results, load_cl_results


class UniformSoup(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()

    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert data_obj is None
        n = len(self.info["particles"])
        merged_particle = linear_combination([1.0 / n] * n, self.info["particles"])
        merged_system = particle2system(merged_particle, ref_system=self.info["ref_system"])
        merged_system.save(f"{self.config['output_dir']['ckpt_dir']}/merged.ckpt")


class TIES(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()
    
    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert data_obj is None

        # define transformation
        def particle2vector(p: ModelParticle) -> torch.Tensor:
            state_dict = p.get_data()
            return torch.nn.utils.parameters_to_vector(state_dict.values())
        def vector2particle(vector: torch.Tensor) -> ModelParticle:
            ref_state_dict = system2particle(self.info["ref_system"]).get_data()
            torch.nn.utils.vector_to_parameters(vector, ref_state_dict.values())
            return ModelParticle(ref_state_dict)

        executor = merging.TIES(
            lamb=self.config["strategy_config"]["lambda"],
            density=self.config["strategy_config"]["density"]
        )
        merged_vector = executor.merge(
            ref_vector=particle2vector(system2particle(self.info["ref_system"])),
            vectors=[particle2vector(p) for p in self.info["particles"]]
        )
        merged_system = particle2system(vector2particle(merged_vector), ref_system=self.info["ref_system"])
        merged_system.save(f"{self.config['output_dir']['ckpt_dir']}/merged.ckpt")


class GreedySoup(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        # self.info = load_exp0_results()
        self.info = load_cl_results("results/exp1/seq-ft")
        self.info["particles"].append(system2particle(self._get_initial_system()))

    def _get_initial_system(self):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"])
        )

    def eval_particle(self, particle: ModelParticle, ds) -> float:
        if getattr(self, "ref_system_for_eval", None) is None:
            self.ref_system_for_eval = self._get_initial_system()
        system = particle2system(particle, ref_system=self.ref_system_for_eval)
        system.eval()
        system.cuda()
        gt, predictions = [], []
        for sample in ds:
            predictions.extend(system.inference([sample["wav"]]))
            gt.append(sample["text"])
        word_error_rate = wer(gt, predictions)
        return -word_error_rate
    
    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert task_name in ["cv-seq", "cv-seq-500"]
        data_obj = data_obj.get_buffer(-1)

        soup_config = {"cache_dir": self.config['output_dir']['log_dir']}
        executor = GreedySoupExecutor(
            soup_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self.eval_particle, ds=data_obj)  # currently depend on cls(data_obj)
        )
        merged_particle = executor.run(self.info["particles"])
        merged_system = particle2system(merged_particle, ref_system=self._get_initial_system())
        merged_system.save(f"{self.config['output_dir']['ckpt_dir']}/merged.ckpt")

    def log(self, x):
        print(f"[GreedySoup]: {x}")


class ModelSwarm(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()

    def eval_particle(self, particle: ModelParticle, ds) -> float:
        system = particle2system(particle, ref_system=self.info["ref_system"])
        system.eval()
        system.cuda()
        gt, predictions = [], []
        for sample in ds:
            predictions.extend(system.inference([sample["wav"]]))
            gt.append(sample["text"])
        word_error_rate = wer(gt, predictions)
        return -word_error_rate
    
    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert task_name in ["cv-seq", "cv-seq-500"]
        data_obj = data_obj.get_buffer(-1)
        swarm_config = copy.deepcopy(self.config["strategy_config"]["swarm"])
        swarm_config["cache_dir"] = f"{self.config['output_dir']['log_dir']}/cache"
        swarm_executor = SwarmExecutor(
            swarm_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self.eval_particle, ds=data_obj)  # currently depend on cls(data_obj)
        )
        merged_particle = swarm_executor.run(self.info["particles"])
        merged_system = particle2system(merged_particle, ref_system=self.info["ref_system"])
        merged_system.save(f"{self.config['output_dir']['ckpt_dir']}/merged.ckpt")
