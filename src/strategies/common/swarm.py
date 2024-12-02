from typing import Type, TypeVar, Generic, Callable
import random
from tqdm import tqdm

from .interface import IParticle


T = TypeVar("T", bound=IParticle)


class SwarmExecutor(Generic[T]):

    config: dict
    linear_operator: Callable[[list[float], list[T]], T]
    utility_function: Callable[[T], float]

    particles: list[T]
    personal_best: list[tuple[T, float]]
    global_best: tuple[T, float]
    global_worst: tuple[T, float]
    velocities: list[T]

    def __init__(self,
        config: dict,
        cls_type: Type[T],
        linear_operator: Callable[[list[float], list[T]], T],
        utility_function: Callable[[T], float]
    ) -> None:
        self.config = config
        self.cls_type = cls_type
        self.linear_operator = linear_operator
        self.utility_function = utility_function

    def _init_search(self, particles: list[T]) -> None:
        assert self.config["N"] >= len(particles), f"N{self.config['N']} should not less than the number of input particles ({len(particles)})!"
        self.log("Initialization...")
        self.particles = []
        self.personal_best = []
        self.global_best, self.global_worst = (None, -2e9), (None, 2e9)
        self.velocities = []
        self.current_iteration = 0
        self.step_length = self.config["step_length"]
        self.patience = 0
        self.personal_patience = [0] * self.config["N"]

        # init location 
        for idx, particle in enumerate(particles):
            particle.cache(self.config["cache_dir"], tag=f"0:loc-{idx}")
            self.particles.append(particle)
        # random interpolation
        for idx in range(len(particles), self.config["N"]):
            tmp = random.sample(particles, 2)
            t = random.random()
            new_particle = self.linear_operator([t, 1 - t], tmp)
            new_particle.cache(self.config["cache_dir"], tag=f"0:loc-{idx}")
            self.particles.append(new_particle)
        assert len(self.particles) == self.config["N"]

        utilities = []
        for particle in tqdm(self.particles):
            utilities.append(self.utility_function(particle))
        self.log(utilities)

        # init personal best, global best/worst
        for idx, (particle, u) in enumerate(zip(self.particles, utilities)):
            particle.cache(self.config["cache_dir"], tag=f"0:best-{idx}")
            self.personal_best.append((particle, u))
            if u > self.global_best[1]:
                self.global_best = (particle, u)
            if u < self.global_worst[1]:
                self.global_worst = (particle, u)
        self.global_best[0].cache(self.config["cache_dir"], tag="0:best")
        self.global_worst[0].cache(self.config["cache_dir"], tag="0:worst")

        # init velocity
        for particle in self.particles:
            tmp = random.sample(particles, 1)[0]
            velocity = self.linear_operator([1, -1], [particle, tmp])
            velocity.cache(self.config["cache_dir"], tag=f"0:v-{idx}")
            self.velocities.append(velocity)

    def _update_velocity_and_location(self, idx: int) -> None:
        # weight randomness
        if self.config["weight_randomness"]:
            r_w = random.uniform(0, 1)
            r_p = random.uniform(0, 1)
            r_s = random.uniform(0, 1)
            r_b = random.uniform(0, 1) # b for bad, repel term weight
        else:
            r_w = 1
            r_p = 1
            r_s = 1
            r_b = 1

        # weight normalize
        self_weight = r_w * self.config["inertia"]
        cognitive_weight = r_p * self.config["cognitive_coeff"]
        social_weight = r_s * self.config["social_coeff"]
        repel_weight = r_b * self.config.get("repel_coeff", 0.0)
        weight_sum = self_weight + cognitive_weight + social_weight + repel_weight
        self_weight = self_weight / weight_sum
        cognitive_weight = cognitive_weight / weight_sum
        social_weight = social_weight / weight_sum
        repel_weight = repel_weight / weight_sum

        # update
        involved_particles = [self.velocities[idx], self.personal_best[idx][0], self.global_best[0], self.global_worst[0], self.particles[idx]]
        coeffs = [self_weight, cognitive_weight, social_weight, -repel_weight, -cognitive_weight-social_weight+repel_weight]
        new_velocity = self.linear_operator(coeffs, involved_particles)
        new_velocity.cache(self.config["cache_dir"], f"{self.current_iteration}:v-{idx}")
        self.velocities[idx] = new_velocity
        new_particle = self.linear_operator([1, self.step_length], [self.particles[idx], new_velocity])
        new_particle.cache(self.config["cache_dir"], f"{self.current_iteration}:loc-{idx}")
        self.particles[idx] = new_particle

    def _run_one_iteration(self):
        # update velocity and location
        for idx in range(self.config["N"]):
            self._update_velocity_and_location(idx)

        utilities = []
        for particle in tqdm(self.particles):
            utilities.append(self.utility_function(particle))
        self.log(utilities)
        
        # update personal best, global best/worst
        self.patience += 1
        for idx, (particle, u) in enumerate(zip(self.particles, utilities)):
            self.personal_patience[idx] += 1
            if u > self.personal_best[idx][1]:
                self.personal_patience[idx] = 0
                self.personal_best[idx] = (particle, u)
            self.personal_best[idx][0].cache(self.config["cache_dir"], tag=f"{self.current_iteration}:best-{idx}")
            if u > self.global_best[1]:
                self.patience = 0
                self.global_best = (particle, u)
            if u < self.global_worst[1]:
                self.global_worst = (particle, u)
        self.global_best[0].cache(self.config["cache_dir"], tag=f"{self.current_iteration}:best")
        self.global_worst[0].cache(self.config["cache_dir"], tag=f"{self.current_iteration}:worst")

        # personal restart
        if self.config.get("personal_patience", 0) > 0:
            for idx in range(self.config["N"]):
                if self.personal_patience[idx] >= self.config["personal_patience"]:
                    self.particles[idx] = self.personal_best[idx][0]
                    self.particles[idx].cache(self.config["cache_dir"], f"{self.current_iteration}:loc-{idx}")
                    self.velocities[idx] = self.cls_type.dummy()
                    self.velocities[idx].cache(self.config["cache_dir"], f"{self.current_iteration}:v-{idx}")              
    
    def run(self, particles: list[T]) -> T:
        self._init_search(particles)
        self.log(f"Iteration {self.current_iteration}:")
        self.log(f"Global best: {self.global_best[1]}.")
        for i in tqdm(range(self.config["iterations"]), desc="Iter"):
            self.current_iteration = i + 1
            self._run_one_iteration()
            self.log(f"Iteration {self.current_iteration}:")
            self.log(f"Global best: {self.global_best[1]}.")
            if self.patience >= self.config["patience"]:  # early stop
                break
        return self.global_best[0]
    
    def log(self, x):
        print(f"[Swarm]: {x}")
