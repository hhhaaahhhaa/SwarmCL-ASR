from tqdm import tqdm
import copy
import yaml
import json
from functools import partial

from src.systems.load import get_system_cls
from src.strategies.base import IStrategy
from src.utils.tool import wer
from .common.swarm import SwarmExecutor
from .swarm_cl.particle import ModelParticle, linear_combination, system2particle, particle2system


def load_exp0_results():
    res = {}
    particles = []
    for accent in ["aus", "eng", "ind", "ire", "sco"]:
        ckpt_path = f"results/exp0/{accent}/ckpt/last.ckpt"
        config = yaml.load(open(f"results/exp0/{accent}/config.yaml", "r"), Loader=yaml.FullLoader)
        system_cls = get_system_cls(config["system_name"])
        system = system_cls.load_from_checkpoint(ckpt_path)
        particles.append(system2particle(system))
        # print(len(particles[-1].get_data().keys()))
    res["particles"] = particles
    res["ref_system"] = system
    return res


class UniformSoup(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()

    def run(self, data_obj):
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
        assert data_obj is None


class GreedySoup(IStrategy):
    def __init__(self, config) -> None:
        self.config = config

        self.info = load_exp0_results()

    def eval_particle(self, particle: ModelParticle, ds) -> float:
        system = particle2system(particle, ref_system=self.info["ref_system"])
        system.eval()
        system.cuda()
        gt, predictions = [], []
        for sample in tqdm(ds):
            predictions.extend(system.inference([sample["wav"]]))
            gt.append(sample["text"])
        word_error_rate = wer(gt, predictions)
        return -word_error_rate
    
    def run(self, data_obj):
        utilities = [self.eval_particle(particle, data_obj._dataset) for particle in self.info["particles"]]
        p_and_u = sorted(list(zip(self.info["particles"], utilities)), key=lambda x: x[1], reverse=True)
        
        soup = [p_and_u[0][0]]
        global_best = p_and_u[0]
        record = {"soup_idx": [0], "utility": [global_best[1]]}
        self.log(f"Iteration 1:")
        self.log(f"Soup indices: {record['soup_idx']}.")
        self.log(f"Global best: {global_best[1]}.")
        for i in range(1, len(p_and_u)):
            merged_particle = linear_combination([1.0 / (len(soup) + 1)] * (len(soup) + 1), [*soup, p_and_u[i][0]])
            u = self.eval_particle(merged_particle, data_obj._dataset)  # currently depend on cls(data_obj)
            if u > global_best[1]:
                soup.append(p_and_u[i][0])
                record["soup_idx"].append(i)
                global_best = (merged_particle, u)
            record["utility"].append(global_best[1])
            self.log(f"Iteration {i+1}:")
            self.log(f"Soup indices: {record['soup_idx']}.")
            self.log(f"Global best: {global_best[1]}.")
        merged_system = particle2system(global_best[0], ref_system=self.info["ref_system"])
        merged_system.save(f"{self.config['output_dir']['ckpt_dir']}/merged.ckpt")
        with open(f"{self.config['output_dir']['log_dir']}/record.json", "w") as f:
            json.dump(record, f, indent=4)

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
        swarm_config = copy.deepcopy(self.config["strategy_config"]["swarm"])
        swarm_config["cache_dir"] = f"{self.config['output_dir']['log_dir']}/cache"
        swarm_executor = SwarmExecutor(
            swarm_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self.eval_particle, ds=data_obj._dataset)  # currently depend on cls(data_obj)
        )
        merged_particle = swarm_executor.run(self.info["particles"])
        merged_system = particle2system(merged_particle, ref_system=self.info["ref_system"])
        merged_system.save(f"{self.config['output_dir']['ckpt_dir']}/merged.ckpt")
