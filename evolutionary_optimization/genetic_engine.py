from typing import List

from evolutionary_optimization.building_blocks import Solution
from evolutionary_optimization.environment_interaction import run_mario


class MetaGeneticAlgorithm:

    def __init__(self, pop_size: int, crossover_rate: float, crossover_points: int, mutation_rate_discrete: float,
                 mutation_rate_continuous: float, current_generation:List[Solution] = None, precision: int = 5):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.crossover_points = crossover_points
        self.mutation_rate_discrete = mutation_rate_discrete
        self.mutation_rate_continuous = mutation_rate_continuous
        self.number_of_elite = int(pop_size * 0.085)
        self.precision = precision

        if current_generation is None:
            self.current_generation: List[Solution] = []
        else:
            self.current_generation = current_generation

    def optimize_network_hyperparameters(self, max_generations=350, NEAT_iterations=50):

        generation_number = 1

        # initialize the first generation

        best_score_all_time = float('-inf')

        is_evolving = True

        while is_evolving:

            if generation_number == max_generations:
                is_evolving = False

            if generation_number > int(max_generations / 2.02):
                self.mutation_rate_continuous /= 4
                self.mutation_rate_discrete /= 4

            best_score_this_generation = float('-inf')

            # TODO: make the current_generation member variable functional and concise

            # the current generation
            for candidate in self.current_generation.solution_list:

                config_path = candidate.config_path
                fitness = run_mario(config_path, NEAT_iterations)
                candidate.fitness = fitness

                if fitness > best_score_this_generation:
                    best_score_this_generation = fitness

            if best_score_this_generation > best_score_all_time:
                best_score_all_time = best_score_this_generation

            next_generation = self.current_generation.set_solution(self.pop_size, self.mutation_rate_discrete,
                                                                   self.mutation_rate_continuous, self.precision)




if __name__ == '__main__':
    pass
