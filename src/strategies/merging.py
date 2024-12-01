from tqdm import tqdm
import copy
import yaml

from src.systems.load import get_system_cls
from src.strategies.base import IStrategy
from .common.swarm import SwarmExecutor
from .swarm_cl.particle import ModelParticle, linear_combination, system2particle, particle2system


def load_exp0_results():
    res = {}
    particles = []
    for accent in ["aus", "eng", "ind", "ire", "sco"]:
        ckpt_path = f"results/exp0/{accent}/ckpt/last.ckpt"
        config = yaml.load(open(f"results/exp0/{accent}/config.yaml", "r"), Loader=yaml.FullLoader)
        system_config = config["config"]
        system_cls = get_system_cls(config["system_name"])
        system = system_cls.load_from_checkpoint(ckpt_path, config=system_config)
        particles.append(system2particle(system))
    res["particles"] = particles
    return res


class UniformSoup(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()

    def run(self, data_obj):
        n = len(self.info["particles"])
        merged_particle = linear_combination([1.0 / n] * n, self.info["particles"])
        merged_system = particle2system(merged_particle)
        merged_system.save(self.config["output_dir"]["ckpt_path"])


class GreedySoup(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()

    def run(self, data_obj):
        n = len(self.info["particles"])
        tmp = linear_combination([1.0 / n] * n, self.info["particles"])
        system = particle2system(tmp)
        system.save(self.config["output_dir"]["ckpt_path"])


class ModelSwarm(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()

    def eval_particle(self, particle: ModelParticle) -> float:
        return 0.0
    
    def run(self, data_obj):
        swarm_config = copy.deepcopy(self.config["swarm"])
        swarm_config["cache_dir"] = "hahaha"
        swarm_executor = SwarmExecutor(
            swarm_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=self.eval_particle
        )
        merged_particle = swarm_executor.run(self.info["particles"])
        merged_system = particle2system(merged_particle)
        merged_system.save(self.config["output_dir"]["ckpt_path"])
