import math
import random
from tqdm import tqdm


class BruteForce:
    def __init__(self, all_solutions, objective_function):
        self.all_solutions = all_solutions
        self.objective_function = objective_function
        self.current_solution = self.all_solutions[0]
        self.best_solution = self.all_solutions[0]
        self.best_score = self.objective_function(self.best_solution)

    def optimize(self):
        for candidate in tqdm(self.all_solutions[1:]):
            candidate_score = self.objective_function(candidate)
            if candidate_score > self.best_score:
                self.best_solution = candidate
                self.best_score = candidate_score
        print(f"Best score={self.best_score}.")

        return self.best_solution, self.best_score


class SimulatedAnnealing:
    def __init__(self, initial_solution, objective_function, neighbor_function,
                 initial_temperature=1000, cooling_rate=0.99, temperature_threshold=1e-3, max_iter=2e9):
        """
        Initialize the Simulated Annealing algorithm.

        :param initial_solution: Initial solution to start with.
        :param objective_function: Function to evaluate the solution quality.
        :param neighbor_function: Function to generate a neighboring solution.
        :param initial_temperature: Starting temperature.
        :param cooling_rate: Rate at which temperature decreases.
        :param temperature_threshold: Minimum temperature to stop the algorithm.
        """
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.objective_function = objective_function
        self.neighbor_function = neighbor_function
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.temperature_threshold = temperature_threshold
        self.best_score = self.objective_function(self.best_solution)

        self.max_iter = max_iter

    def _acceptance_probability(self, current_score, neighbor_score):
        """
        Calculate the acceptance probability for a worse solution.

        :param current_score: Objective function score of the current solution.
        :param neighbor_score: Objective function score of the neighbor solution.
        :return: Acceptance probability.
        """
        if neighbor_score > current_score:
            return 1.0
        return math.exp((neighbor_score - current_score) / self.temperature)

    def optimize(self):
        """
        Perform the optimization using the simulated annealing algorithm.

        :return: Best solution and its objective function score.
        """
        n_iter = 0
        while self.temperature > self.temperature_threshold and n_iter < self.max_iter:
            neighbor = self.neighbor_function(self.current_solution)
            current_score = self.objective_function(self.current_solution)
            neighbor_score = self.objective_function(neighbor)

            if random.random() < self._acceptance_probability(current_score, neighbor_score):
                self.current_solution = neighbor

            # Update the best solution if the neighbor is better
            if neighbor_score > self.best_score:
                self.best_solution = neighbor
                self.best_score = neighbor_score

            # Cool down the temperature
            self.temperature *= self.cooling_rate
            n_iter += 1
            print(f"Iter={n_iter}, best score={self.best_score}.")

        return self.best_solution, self.best_score


if __name__ == "__main__":
    # Example usage
    def objective_function(x):
        return x ** 2  # Minimize the square function

    def neighbor_function(x):
        return x + random.uniform(-1, 1)  # Randomly move to a neighbor

    # Initialize the algorithm
    sa = SimulatedAnnealing(initial_solution=10, 
                            objective_function=objective_function,
                            neighbor_function=neighbor_function)

    # Run optimization
    best_solution, best_score = sa.optimize()
    print("Best solution:", best_solution)
    print("Best score:", best_score)
