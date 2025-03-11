import os
import pickle

from evolutionary_optimization.building_blocks import IndividualSolution
from evolutionary_optimization.environment_interaction import run_mario
from evolutionary_optimization.utils import get_hamming_neighbours

def extract_weights_and_biases(solution_path):

    weights_arr = []
    biases_arr = []

    with open(solution_path, 'rb') as winner:
        genome = pickle.load(winner)

    print(f"genome = \n{genome}\n")
    weights = {key: conn.weight for key, conn in genome.connections.items()}

    print(f"Weights: ")
    for conn, weight in weights.items():
        print(f"Connection {conn}: Weight {weight}")
        weights_arr.append(weight)

    print(f"Biases: ")
    biases = {key: node.bias for key, node in genome.nodes.items()}
    for node, bias in biases.items():
        print(f"Node: {node} with bias {bias}")
        biases_arr.append(biases)

    return weights, biases, weights_arr, biases_arr


class HillClimbing:

    def __init__(self, current_network_path, config_path='config75', precision=5):

        if current_network_path is not None:
            self.network_path = current_network_path
        else:
            print("You must specify the network path for which the search is performed")
            return

        self.config_path = config_path
        self.precision = precision

    # start: 861,6 distance avg over 10 runs

    def improve_solution(self, current_best, num_runs=10):


        num_until_improvements = []
        best_distance = current_best
        best_bitstring = None
        local = True

        while local:

            found_improvement = False
            new_candidate = IndividualSolution()
            neighbours = get_hamming_neighbours(best_distance, self.num_samples)

            if new_fitness > best_distance.fitness:
                best_distance = new_candidate
                num_until_improvements.append(i + 1)
                found_improvement = True
                break  #

            if not found_improvement:
                local = False

        print(f"best_config = {best_distance.config_path}")
        return new_network


if __name__ == '__main__':

    current_path = "../models/winner_config75_copy.pkl"
    current_path1 = "modified_network.pkl"

    weights, biases, weights_list, biases_list = extract_weights_and_biases(current_path1)
    hill_climbing = HillClimbing(current_path)
    hill_climbing.improve_solution(current_best=861.6)


