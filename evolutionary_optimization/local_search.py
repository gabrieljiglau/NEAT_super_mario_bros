import os
import pickle

from evolutionary_optimization.building_blocks import IndividualSolution
from evolutionary_optimization.environment_interaction import run_mario
from evolutionary_optimization.utils import get_hamming_neighbours


class HillClimbing:

    def __init__(self, current_solution:IndividualSolution, mutate_discrete, mutate_continuous,
                 precision=5, num_samples=30):

        if current_solution is not None:
            self.current_solution = current_solution
        else:
            print("You must specify a starting config")
            return

        self.mutate_discrete = mutate_discrete
        self.mutate_continuous = mutate_continuous
        self.precision = precision
        self.num_samples = num_samples

    """
    wrrrong; the search will be done on the neural networks weights and biases
    """

    def improve_solution(self, NEAT_generations=20):

        num_until_improvements = []
        best = self.current_solution
        local = True

        """
        salvez in pickle: config-ul nou, num_improvements(,adica de câte ori s-a putut îmbunătăți soluția curentă)
        scriu in fisier: după cate sample-uri s-a produs imbunătățirea
        """

        while local:

            found_improvement = False
            new_candidate = IndividualSolution()
            neighbours = get_hamming_neighbours(best, self.num_samples)

            for i in range(self.num_samples):
                new_candidate.set_initial_solution(self.mutate_discrete, self.mutate_continuous, self.precision,
                                                   local, neighbours[i])

                new_fitness = run_mario(new_candidate.config_path, NEAT_generations, local)
                if new_fitness > best.fitness:
                    best = new_candidate
                    num_until_improvements.append(i + 1)
                    found_improvement = True
                    break  #

            if not found_improvement:
                local = False

        print(f"best_config = {best.config_path}")
        return best.config_path


if __name__ == '__main__':
    pass