import os
import pickle
from evolutionary_optimization.environment_interaction import run_mario
from evolutionary_optimization.run_winner_configuration import run_neat_iteratively
from evolutionary_optimization.utils import get_hamming_neighbours, modify_network, build_continuous_parameters, \
    extract_weights_and_biases


class HillClimbing:

    def __init__(self, current_network_path, config_path, precision=5):

        if current_network_path is not None:
            self.network_path = current_network_path
        else:
            print("You must specify the network path !!")
            return

        self.config_path = config_path
        self.precision = precision


    def improve_solution(self, lower_bound=-3, upper_bound=3, num_runs=10,
                         file_tracker='distance_logger_HC.txt', checkpoint_file='../checkpoints/hc_checkpoint.pkl'):

        """
        inițial dau ca parametru calea către rețea,
        iar apoi, mă interesează doar weight-urile sub forma de bitstring, gasite in iteratia trecuta
        """

        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.network_path = saved_data['network']
                num_improvements = saved_data['num_improvements']
                weights_and_biases = saved_data['weights_and_biases']
                local = saved_data['local']
                best_distance = saved_data['best_distance']
                print('successfully opened checkpoint')
        else:
            num_improvements = 0
            float_values = extract_weights_and_biases(self.network_path)  # the actual values
            weights_and_biases = build_continuous_parameters(float_values, lower_bound, upper_bound)
            best_distance = 861.6  # start: 861.6 avg distance in 10 runs
            local = True

        best_neighbour = None

        # I want to run hill climbing improvement by improvement
        while local:

            found_improvement = False
            neighbours = get_hamming_neighbours(weights_and_biases)
            print(neighbours[1])

            """
            for neighbour_bitstring in neighbours:

                new_network = modify_network(neighbour_bitstring, lower_bound, upper_bound)
                new_avg_distance = run_neat_iteratively(new_network, num_runs)

                if new_avg_distance > best_distance:
                    best_neighbour = neighbour_bitstring
                    found_improvement = True
                    num_improvements += 1

            if not found_improvement:
                local = False

            try:
                with open(file_tracker, 'a') as f:
                    f.write(f"in iteration {num_improvements + 1} the best distance is {best_distance}\n")
            except IOError:
                print(f"Error when trying to write to file {file_tracker}")

            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'network': self.network_path,
                                'num_improvements': num_improvements,
                                 'weights_and_biases': best_neighbour,
                                 'best_distance': best_distance,
                                 'local': local}, f)
            except IOError:
                print(f"Error when writing to pickle {checkpoint_file}")

            break
        """
        return self.network_path


if __name__ == '__main__':

    config_path = "../configs/config75"
    network_path = "../models/winner_config75_original.pkl"
    current_path1 = "modified_network.pkl"

    hill_climbing = HillClimbing(network_path, config_path, precision=5)
    hill_climbing.improve_solution()
