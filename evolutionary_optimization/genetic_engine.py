import os
import pickle
from typing import List, Dict
from evolutionary_optimization.building_blocks import Solution, IndividualSolution
from evolutionary_optimization.environment_interaction import run_mario
from evolutionary_optimization.utils import Randomizer, get_bit_num


def select_elite(solution_list: List[IndividualSolution], number_of_elites: int):
    sorted_generation = sorted(solution_list, key=lambda individual: individual.fitness, reverse=True)
    return sorted_generation[0:number_of_elites], sorted_generation


def crossover_parents(parent1: IndividualSolution, parent2: IndividualSolution,
                      crossover_points, crossover_rate) -> List[IndividualSolution] | None:
    offspring_list = []

    if len(parent1.parameters) != len(parent2.parameters):
        print(f"The two selected parents for crossover do not have the same length; p1: {parent1}, p2:{parent2} ")
        return None

    new_bitstring1 = list(parent1.big_bitstring)
    new_bitstring2 = list(parent2.big_bitstring)

    did_crossover = False
    for _ in range(crossover_points):
        if Randomizer.next_double() < crossover_rate:

            did_crossover = True
            selected_gene_idx = Randomizer.get_int_between(0, len(parent1.parameters) - 1)
            selected_gene = parent1.parameters[selected_gene_idx]

            gene_start = sum(
                get_bit_num(param.lower_bound, param.upper_bound, param.precision)
                for param in parent1.parameters[:selected_gene_idx]
            )

            gene_bit_count = get_bit_num(
                selected_gene.lower_bound, selected_gene.upper_bound, selected_gene.precision
            )

            gene_end = gene_start + gene_bit_count

            bitstring1 = parent1.big_bitstring[gene_start:gene_end]
            bitstring2 = parent2.big_bitstring[gene_start:gene_end]

            if len(bitstring1) != len(bitstring2):
                print(f"The selected parents mismatched on selected gene for crossover;  "
                      f"b1:{bitstring1}, b2:{bitstring2}")
                return None

            crossover_point = Randomizer.get_int_between(0, len(bitstring1) - 1)

            new_bitstring1[gene_start + crossover_point:gene_end] = \
                parent2.big_bitstring[gene_start + crossover_point:gene_end]

            new_bitstring2[gene_start + crossover_point:gene_end] = \
                parent1.big_bitstring[gene_start + crossover_point:gene_end]

    child1 = IndividualSolution(''.join(new_bitstring1))
    child2 = IndividualSolution(''.join(new_bitstring2))

    if did_crossover:
        child1.decode_individual_solution()
        child2.decode_individual_solution()

    offspring_list.append(child1)
    offspring_list.append(child2)

    return offspring_list


class MetaGeneticAlgorithm:

    def __init__(self, pop_size: int, crossover_rate: float, crossover_points: int, mutation_rate_discrete: float,
                 mutation_rate_continuous: float, current_generation: Solution = None, precision: int = 5):
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

    def optimize_network_hyperparameters(self, max_generations=350, NEAT_generations=50,
                                         file_tracker='../logging/fitness_logger_meta_GA.txt', checkpoint_file='checkpoint.pkl'):

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                print('successfully opened the checkpoint')
                saved_data = pickle.load(f)
                self.current_generation = saved_data['current_generation']
                generation_number = saved_data['generation_number']
                best_score_all_time = saved_data['best_score_all_time']
                best_individual = saved_data['best_individual']
                IndividualSolution.individual_id = saved_data['individual_id']
                print(f"last_config path = {IndividualSolution.individual_id}")
        else:
            generation_number = 1
            best_score_all_time = float('-inf')
            best_individual = None

        winners_percentage = 0
        winners = 0
        is_evolving = True
        total_fitness = 0
        while is_evolving:

            print(f"Now running generation number {generation_number}")

            if generation_number == max_generations:
                is_evolving = False

            if generation_number > int(max_generations / 2.02):
                self.mutation_rate_continuous /= 4
                self.mutation_rate_discrete /= 4

            best_score_this_generation = float('-inf')

            best_candidate = None
            # the current generation
            index = 1

            print(f"Configs used:")
            for candidate in self.current_generation.solution_list:
                print(f"{candidate.config_path}")

            for candidate in self.current_generation.solution_list:
                print(f"now evaluating candidate: {index}")
                config_path = candidate.config_path
                fitness = run_mario(config_path, NEAT_generations, optimizing=True)
                if fitness > 10000:
                    winners += 1
                total_fitness += fitness
                candidate.fitness = fitness

                if fitness > best_score_this_generation:
                    best_score_this_generation = fitness
                    best_candidate = candidate

                index += 1

            if best_score_this_generation > best_score_all_time:
                best_score_all_time = best_score_this_generation
                best_individual = best_candidate

            next_generation = Solution()
            elite_individuals, sorted_population = select_elite(self.current_generation.solution_list,
                                                                self.number_of_elite)
            elite_configs = [elite.config_path for elite in elite_individuals]
            next_generation.solution_list.extend(elite_individuals)
            """
            add the elites to the next generation and complete the rest of the population through selection
            """

            while len(next_generation.solution_list) < self.pop_size:
                selected_parents = self.select_candidates(sorted_population)
                offsprings = crossover_parents(selected_parents[0], selected_parents[1],
                                               self.crossover_points, self.crossover_rate)

                child1, child2 = offsprings[0], offsprings[1]

                new_bitstring1 = IndividualSolution.mutate_individual_solution(child1, self.mutation_rate_discrete,
                                                                               self.mutation_rate_continuous)
                new_bitstring2 = IndividualSolution.mutate_individual_solution(child2, self.mutation_rate_discrete,
                                                                               self.mutation_rate_continuous)

                child1.big_bitstring = new_bitstring1
                child2.big_bitstring = new_bitstring2

                next_generation.solution_list.append(child1)
                next_generation.solution_list.append(child2)

            for individual in self.current_generation.solution_list:
                if individual.config_path not in elite_configs:
                    individual.delete_config()

            self.current_generation = next_generation
            print(f"best_evaluation {best_score_all_time} in generation {generation_number}")

            generation_number += 1

            try:
                with open(file_tracker, 'a') as f:
                    f.write(f"best_evaluation {best_score_all_time} in generation {generation_number}; ")
                    f.write(f"mean fitness {float(total_fitness/self.pop_size)}; ")
                    f.write(f"win percentage {float(winners/self.pop_size)} \n")
            except IOError:
                print(f"Error when trying to write to file {file_tracker}")

            print(f"saving generation number {generation_number} to {checkpoint_file}")
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'current_generation': self.current_generation,
                             'generation_number': generation_number,
                             'best_score_all_time': best_score_all_time,
                             'best_individual': best_individual,
                             'individual_id': IndividualSolution.individual_id}, f)  # Save last used ID

        return best_individual.config_path

    def select_candidates(self, sorted_population: List[IndividualSolution]) -> List[IndividualSolution]:
        """
        rank-based roulette-wheel selection; it assigns the largest probability for the highest ranked individual,
        and it ensures that even low-fitness individuals have a non-zero chance of being selected
        :return: two candidates
        """

        rank_map: Dict[IndividualSolution, int] = {sorted_population[i]: i + 1 for i in range(self.pop_size)}

        selection_probabilities: List[float] = []
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

    population_size = 30
    cx_rate = 0.8
    cx_points = 30
    mutate_rate_discrete = 0.004
    mutate_rate_continuous = 0.006

    """
    with open('checkpoint.pkl', 'rb') as f:
        checkpoint = pickle.load(f)
        curr_generation = checkpoint['current_generation']

    for candidate in curr_generation.solution_list:
        print(f"{candidate.config_path} has fitness -> {candidate.fitness}")
    """

    """
    print(f"The supposed configs")
    for candidate in curr_generation.solution_list:
        print(f"{candidate.config_path}")
    """

    meta_genetic_algorithm = MetaGeneticAlgorithm(population_size, cx_rate, cx_points,
                                                  mutate_rate_discrete, mutate_rate_continuous)
    meta_genetic_algorithm.optimize_network_hyperparameters(30, 20)
