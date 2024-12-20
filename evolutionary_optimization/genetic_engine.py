from typing import List, Dict
from evolutionary_optimization.building_blocks import Solution, IndividualSolution
from evolutionary_optimization.environment_interaction import run_mario
from evolutionary_optimization.utils import Randomizer


def select_elite(solution_list: List[IndividualSolution], number_of_elites:int):
    sorted_generation = sorted(solution_list, key=lambda individual: individual.fitness, reverse=True)
    return sorted_generation[0:number_of_elites], sorted_generation

def crossover_parents(parent1: IndividualSolution, parent2: IndividualSolution) -> List[IndividualSolution]:
    # TODO:add crossover here
    # parent_list = []
    pass


class MetaGeneticAlgorithm:

    def __init__(self, pop_size: int, crossover_rate: float, crossover_points: int, mutation_rate_discrete: float,
                 mutation_rate_continuous: float, current_generation:Solution = None, precision: int = 5):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.crossover_points = crossover_points
        self.mutation_rate_discrete = mutation_rate_discrete
        self.mutation_rate_continuous = mutation_rate_continuous
        self.number_of_elite = int(pop_size * 0.085)
        self.precision = precision

        if current_generation is None:
            self.current_generation = Solution.create_first_generation(
                pop_size=pop_size,
                mutate_discrete=mutation_rate_discrete,
                mutate_continuous=mutation_rate_continuous,
                precision=precision)
        else:
            self.current_generation = current_generation

    def optimize_network_hyperparameters(self, max_generations=350, NEAT_iterations=50):

        generation_number = 1
        best_score_all_time = float('-inf')
        is_evolving = True
        best_individual = None

        while is_evolving:

            if generation_number == max_generations:
                is_evolving = False

            if generation_number > int(max_generations / 2.02):
                self.mutation_rate_continuous /= 4
                self.mutation_rate_discrete /= 4

            best_score_this_generation = float('-inf')

            best_candidate = None
            # the current generation
            for candidate in self.current_generation.solution_list:

                config_path = candidate.config_path
                fitness = run_mario(config_path, NEAT_iterations)
                candidate.fitness = fitness

                if fitness > best_score_this_generation:
                    best_score_this_generation = fitness
                    best_candidate = candidate

            if best_score_this_generation > best_score_all_time:
                best_score_all_time = best_score_this_generation
                best_individual = best_candidate

            next_generation = Solution()
            elite_individuals, sorted_population = select_elite(self.current_generation.solution_list,
                                                                self.number_of_elite)
            next_generation.solution_list.extend(elite_individuals)
            """
            add the elites to the next generation and complete the rest of the population through selection
            """

            index = self.number_of_elite
            while len(next_generation.solution_list) < self.pop_size:

                selected_parents = self.select_candidates(sorted_population)
                child1 = Solution()
                child2 = Solution()

                if Randomizer.next_double() < self.crossover_rate:
                    # TODO:add crossover here
                    pass
                else:
                    child1 = selected_parents[0]
                    child2 = selected_parents[1]

                new_bitstring1 = IndividualSolution.mutate_individual_solution(child1, self.mutation_rate_discrete,
                                                                               self.mutation_rate_continuous)
                new_bitstring2 = IndividualSolution.mutate_individual_solution(child2, self.mutation_rate_discrete,
                                                                               self.mutation_rate_continuous)

                child1.big_bitstring = new_bitstring1
                child2.big_bitstring = new_bitstring2

                next_generation.solution_list[index] = child1
                index += 1
                next_generation.solution_list[index] = child2

            for individual in self.current_generation.solution_list:
                if not individual.fitness == best_score_this_generation:
                    individual.delete_config()

            self.current_generation = next_generation
            print(f"best_evaluation {best_score_all_time} in generation {generation_number}")
            generation_number += 1

        return best_individual.config_path

    def select_candidates(self, sorted_population: List[IndividualSolution]) -> List[IndividualSolution]:
        """
        rank-based roulette-wheel selection; it assigns the largest probability for the highest ranked individual,
        and it ensures that even low-fitness individuals have a non-zero chance of being selected
        :return: two candidates
        """

        rank_map:Dict[IndividualSolution, int] = {sorted_population[i]: i+1 for i in range(self.pop_size)}

        selection_probabilities:List[float] = []
        total_probability = 0.0
        for individual in self.current_generation.solution_list:
            rank = rank_map.get(individual)
            probability = 2.0 * (self.pop_size - rank + 1) / (self.pop_size * (self.pop_size + 1))
            selection_probabilities.append(probability)
            total_probability += probability

        # normalize the probabilities to add to 1
        for i in range(self.pop_size):
            selection_probabilities[i] = selection_probabilities[i] / total_probability

        selected_candidates: List[IndividualSolution] = []
        for candidate_index in range(2):
            generated_value = Randomizer.next_double()
            cumulative_probability = 0
            selected_individual_index = 0

            for i in range(self.pop_size):
                cumulative_probability += selection_probabilities[i]
                if cumulative_probability > generated_value:
                    selected_individual_index = i
                    break

            selected_candidates.append(self.current_generation.solution_list[selected_individual_index])

        return selected_candidates


if __name__ == '__main__':
    pass
