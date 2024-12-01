from tqdm import tqdm
import copy

from one import train_one_task
from src.strategies.base import IStrategy
from ..common.swarm import SwarmExecutor
from .particle import ModelParticle, linear_combination, system2particle, particle2system
from .utility import WERUtility


class SwarmCLStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.utility = WERUtility()
    
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
        swarm_executor.run([])
