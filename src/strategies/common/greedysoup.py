import os
import numpy as np
from typing import Type, TypeVar, Generic, Callable
from tqdm import tqdm
import json

from .interface import IParticle


T = TypeVar("T", bound=IParticle)


class GreedySoupExecutor(Generic[T]):

    config: dict
    linear_operator: Callable[[list[float], list[T]], T]
    utility_function: Callable[[T], float]

    def __init__(self,
        config: dict,
        cls_type: Type[T],
        linear_operator: Callable[[list[float], list[T]], T],
        utility_function: Callable[[T], float],
        verbose=True
    ) -> None:
        self.config = config
        self.cls_type = cls_type
        self.linear_operator = linear_operator
        self.utility_function = utility_function
        self.verbose = verbose

    def _init_search(self, particles: list[T], record={}) -> list[tuple[T, float]]:
        self.log("Initialization...")
        utilities = [self.utility_function(particle) for particle in particles]
        record["sort_idx"] = np.argsort(np.array(utilities)).tolist()[::-1]
        p_and_u = sorted(list(zip(particles, utilities)), key=lambda x: x[1], reverse=True)
        return p_and_u
    
    def run(self, particles: list[T], return_record=False) -> T:
        record = {}
        p_and_u = self._init_search(particles, record)

        soup = []
        global_best = (None, -2e9)
        record.update({"soup_idx": [], "utility": []})
        for i in tqdm(range(len(particles))):
            souped_particle = self.linear_operator([1.0 / (len(soup) + 1)] * (len(soup) + 1), [*soup, p_and_u[i][0]])
            u = self.utility_function(souped_particle)
            if u > global_best[1]:
                soup.append(p_and_u[i][0])
                record["soup_idx"].append(i)
                global_best = (souped_particle, u)
            record["utility"].append(global_best[1])
        self.log(f"Sorted indices: {record['sort_idx']}.")
        self.log(f"Soup indices: {record['soup_idx']}.")
        self.log(f"Global best: {global_best[1]}.")

        os.makedirs(self.config['cache_dir'], exist_ok=True)
        with open(f"{self.config['cache_dir']}/record.json", "w") as f:
            json.dump(record, f, indent=4)
        
        if return_record:
            return global_best[0], record
        return global_best[0]
    
    def log(self, x):
        if self.verbose:
            print(f"[GreedySoup]: {x}")
