import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset
from tqdm import tqdm
import copy
import yaml
import json
from functools import partial
import random
import logging

from one import load_system
from src.strategies.base import IStrategy
from src.tasks.utils import TaskSequence
from src.utils.tool import wer
from .common.greedysoup import GreedySoupExecutor
from .common.particle import ModelParticle, linear_combination, system2particle, particle2system
from .common.utils import load_exp0_results, load_cl_results, load_exp0_results_long
from .common.sa import SimulatedAnnealing, BruteForce


class Ingredient(object):
    def __init__(self, particle, tids, utility, n_words):
        self.particle = particle
        self.tids = tids
        self.utility = utility  # Define value as -WER
        self.n_words = n_words  # we need the weight to merge ingredients without recalculating


class MultiModelStrategy(IStrategy):

    def __init__(self, config) -> None:
        self.config = config

        # self.info = load_exp0_results()
        self.info, _ = load_exp0_results_long()
    
    def _get_initial_system(self):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"])
        )

    def run(self, data_obj):
        _, data_obj = data_obj
        tid2gid = {"use_raw_path": True}
        for tid, task_name in enumerate(data_obj.task_names):
            tid2gid[task_name] = self.info["raw_paths"][tid]
        
        dir = self.config['output_dir']['ckpt_dir']
        with open (f"{dir}/info.json", "w") as f:
            json.dump(tid2gid, f, indent=4)


class GeneralCGreedySoup(IStrategy):

    soup: list[Ingredient]

    def __init__(self, config) -> None:
        self.config = config
        self.soup = []
        self._prev_search_result = None
        self._buffer = []
        logging.basicConfig(filename=f"{self.config['output_dir']['log_dir']}/log.log", level=logging.INFO)

        # self.info = load_exp0_results()
        self.info, self.particle_getter = load_exp0_results_long()
        # self.info = load_cl_results("results/exp1/seq-ft")
        # self.info["particles"].append(system2particle(self._get_initial_system()))

    def _get_buffer(self, tids: list[int]):
        return ConcatDataset([self._buffer[x] for x in tids])

    def _get_initial_system(self):
        return load_system(
            system_name=self.config["strategy_config"]["system_name"],
            system_config=copy.deepcopy(self.config["system_config"])
        )

    def _eval_particle(self, particle: ModelParticle, ds) -> float:
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
    
    # simulated annealing
    def _neighbor_function(self, merged_soup_ids: list[int]) -> list[int]:
        assert len(merged_soup_ids) >= 2
        n = len(self.soup)
        if n == 2:
            return merged_soup_ids
        
        while 1:
            current_subset = set(merged_soup_ids)  # Ensure it's a set
            full_set = set(list(range(n)))  # Ensure it's a set

            action = random.choice(["add", "remove", "replace"])
            if action == "add":
                # Add an element not in the subset
                candidates = full_set - current_subset
                if candidates:
                    current_subset.add(random.choice(list(candidates)))

            elif action == "remove":
                # Remove an element from the subset
                if len(current_subset) > 2:
                    current_subset.remove(random.choice(list(current_subset)))

            elif action == "replace":
                # Replace an element in the subset with one not in it
                if current_subset and (full_set - current_subset):
                    element_to_remove = random.choice(list(current_subset))
                    current_subset.remove(element_to_remove)
                    candidates = full_set - current_subset
                    current_subset.add(random.choice(list(candidates)))
            if current_subset != set(merged_soup_ids):
                break

        return list(current_subset)

    def _objective_function(self, merged_soup_ids: list[int]) -> float:
        particles, merged_tids = [], []
        original_utility, denom = 0, 0
        for x in merged_soup_ids:
            particles.append(self.soup[x].particle)
            merged_tids.extend(self.soup[x].tids)
            original_utility += self.soup[x].utility * self.soup[x].n_words
            denom += self.soup[x].n_words
        original_utility = original_utility / denom

        # run merging
        soup_config = {"cache_dir": f"{self.config['output_dir']['log_dir']}/tmp"}
        executor = GreedySoupExecutor(
            soup_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self._eval_particle, ds=self._get_buffer(merged_tids)),
            verbose=False
        )
        _, record = executor.run(particles, return_record=True)
        merged_utility = record["utility"][-1]
        return (merged_utility - original_utility) * denom  # objective is the difference of editdistance before/after merging

    # merging
    def _search_subset_merge(self):
        n = len(self.soup)

        # Brute force
        from itertools import combinations
        def subsets_greater_than_one(n):
            """
            Generate all subsets of {0, 1, ..., n-1} with size > 1.

            :param n: An integer
            :return: A list of tuples, each representing a subset
            """
            elements = list(range(n))
            result = []
            for size in range(2, len(elements) + 1):
                result.extend(combinations(elements, size))
            return result
        all_solutions = subsets_greater_than_one(n)
        optimizer = BruteForce(
            all_solutions=all_solutions,
            objective_function=self._objective_function,
        )

        # Simulated annealing
        # optimizer = SimulatedAnnealing(
        #     initial_solution=random.sample(list(range(n)), 2),
        #     objective_function=self._objective_function,
        #     neighbor_function=self._neighbor_function,
        #     initial_temperature=1,
        #     cooling_rate=0.99,
        #     temperature_threshold=1e-3,
        #     max_iter=1 if n == 2 else min(2 ** n - n - 2, self.config["strategy_config"]["sa_iter"])
        # )

        best_solution, gain = optimizer.optimize()
        logging.info(f"Best solution: {best_solution}")
        logging.info(f"Gain: {gain}")
        return best_solution, gain

    def _perform_merge(self):  # run merging on the best solution
        print("========== Merging ==========")
        best_solution, gain = self._prev_search_result
        logging.info(f"Merge groups {best_solution}...")

        new_soup = []
        particles, merged_tids, original_utility, denom = [], [], 0, 0
        for id in range(len(self.soup)):
            if id not in best_solution:
                new_soup.append(self.soup[id])
            else:
                particles.append(self.soup[id].particle)
                merged_tids.extend(self.soup[id].tids)
                original_utility += self.soup[id].utility * self.soup[id].n_words
                denom += self.soup[id].n_words
        original_utility = original_utility / denom

        # run merging again since we do not save anything during the optimization
        soup_config = {"cache_dir": f"{self.config['output_dir']['log_dir']}/tmp"}
        executor = GreedySoupExecutor(
            soup_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self._eval_particle, ds=self._get_buffer(merged_tids))
        )
        merged_particle, record = executor.run(particles, return_record=True)
        merged_utility = record["utility"][-1]
        # print("Matching: ", original_utility * denom + gain, merged_utility * denom)  # should be equal

        # update soup
        new_soup.append(
            Ingredient(
                particle=merged_particle,
                tids=merged_tids,
                utility=merged_utility,
                n_words=denom
            )
        )
        self.soup = new_soup
        self._prev_search_result = None

    # run
    def _load_start(self, tid, data_obj):
        if len(self.soup) == self.config["strategy_config"]["max_size"]:  # max size achieved, force merge
            logging.info("Max size achieved")
            if self._prev_search_result is None:
                self._prev_search_result = self._search_subset_merge()
            self._perform_merge()

    def _integrate(self, tid, data_obj):
        # eval and create ingredient
        new_particle = self.particle_getter(tid)
        if getattr(self, "ref_system_for_eval", None) is None:
            self.ref_system_for_eval = self._get_initial_system()
        system = particle2system(new_particle, ref_system=self.ref_system_for_eval)
        system.eval()
        system.cuda()
        gt, predictions = [], []
        n_words = 0
        ds = data_obj.tasks[tid].val_dataset()
        ds = Subset(ds, indices=list(range(self.config["strategy_config"]["n_val"])))
        for sample in ds:
            predictions.extend(system.inference([sample["wav"]]))
            gt.append(sample["text"])
            n_words += len(sample["text"].split(" "))
        word_error_rate = wer(gt, predictions)
        self.soup.append(Ingredient(new_particle, [tid], -word_error_rate, n_words))
        self._buffer.append(ds)

        if len(self.soup) == 1:
            return
        best_solution, gain = self._search_subset_merge()
        self._prev_search_result = (best_solution, gain)
        if gain > 0:  # improve if merged
            logging.info("Improved")
            self._perform_merge()

    def run(self, data_obj):
        task_name, data_obj = data_obj
        assert isinstance(data_obj, TaskSequence)
        for tid, task_name in enumerate(data_obj.task_names):
            logging.info(f"Task {tid}: {task_name}")
            self._load_start(tid, data_obj)
            self._integrate(tid, data_obj)

            logging.info("Groups:")
            for j, ingred in enumerate(self.soup):
                logging.info(f"{j}: {ingred.tids}")
            logging.info("")

        self._save_to_group(data_obj.task_names)

        # final merge
        n = len(data_obj.task_names)
        soup_config = {"cache_dir": self.config['output_dir']['log_dir']}
        executor = GreedySoupExecutor(
            soup_config,
            cls_type=ModelParticle,
            linear_operator=linear_combination,
            utility_function=partial(self._eval_particle, ds=self._get_buffer(list(range(n)))),
            verbose=False
        )
        particles = [ingred.particle for ingred in self.soup]
        merged_particle = executor.run(particles)
        merged_system = particle2system(merged_particle, ref_system=self._get_initial_system())
        merged_system.save(f"{self.config['output_dir']['ckpt_dir']}/merged.ckpt")

    def _save_to_group(self, tnames: list[str]):
        dir = self.config['output_dir']['ckpt_dir']
        tid2gid = {}
        for gid, ingred in enumerate(self.soup):
            for tid in ingred.tids:
                tid2gid[tnames[tid]] = str(gid)
            system = particle2system(ingred.particle, ref_system=self.ref_system_for_eval)
            system.save(f"{dir}/{str(gid)}.ckpt")
        with open (f"{dir}/info.json", "w") as f:
            json.dump(tid2gid, f, indent=4)


class GeneralCUniformSoup(GeneralCGreedySoup):
    def _objective_function(self, merged_soup_ids: list[int]) -> float:
        particles, merged_tids = [], []
        original_utility, denom = 0, 0
        for x in merged_soup_ids:
            particles.append(self.soup[x].particle)
            merged_tids.extend(self.soup[x].tids)
            original_utility += self.soup[x].utility * self.soup[x].n_words
            denom += self.soup[x].n_words
        original_utility = original_utility / denom

        # run merging
        merged_particle = linear_combination([1 / len(particles)] * len(particles), particles)
        merged_utility = self._eval_particle(merged_particle, ds=self._get_buffer(merged_tids))
        return (merged_utility - original_utility) * denom  # objective is the difference of editdistance before/after merging
    
    def _perform_merge(self):  # run merging on the best solution
        print("========== Merging ==========")
        best_solution, gain = self._prev_search_result
        logging.info(f"Merge groups {best_solution}...")

        new_soup = []
        particles, merged_tids, original_utility, denom = [], [], 0, 0
        for id in range(len(self.soup)):
            if id not in best_solution:
                new_soup.append(self.soup[id])
            else:
                particles.append(self.soup[id].particle)
                merged_tids.extend(self.soup[id].tids)
                original_utility += self.soup[id].utility * self.soup[id].n_words
                denom += self.soup[id].n_words
        original_utility = original_utility / denom

        # run merging again since we do not save anything during the optimization
        merged_particle = linear_combination([1 / len(particles)] * len(particles), particles)
        merged_utility = self._eval_particle(merged_particle, ds=self._get_buffer(merged_tids))
        # print("Matching: ", original_utility * denom + gain, merged_utility * denom)  # should be equal

        # update soup
        new_soup.append(
            Ingredient(
                particle=merged_particle,
                tids=merged_tids,
                utility=merged_utility,
                n_words=denom
            )
        )
        self.soup = new_soup
        self._prev_search_result = None
