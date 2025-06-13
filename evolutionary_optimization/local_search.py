import os
import pickle
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

        with open(self.network_path, 'rb') as f:
            self.genome = pickle.load(f)


    def improve_solution(self, lower_bound=-3, upper_bound=3, num_runs=20,
                         file_tracker='distance_logger_HC.txt', checkpoint_file='../checkpoints/hc_checkpoint.pkl'):

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

            best_distance = run_neat_iteratively(self.network_path, 'config75', training=True)
            print(f"Initial distance: {best_distance}")
            local = True

        if not local:
            print('Cannot hill-climb further')
            return

        best_neighbour = None
        print(f"starting distance: {best_distance}")
        print(f"now at improvement {num_improvements}")

        # I want to run hill climbing improvement by improvement
        i = 1
        while local:

            found_improvement = False
            neighbours = get_hamming_neighbours(weights_and_biases, sampling=True)
            worst_distance = best_distance

            # when doing worst_improvement, I can just sample 30 neighbours
            print(f"another improvement found")
            for neighbour_bitstring in neighbours:

                i += 1
                new_network = modify_network(self.network_path, neighbour_bitstring, lower_bound, upper_bound,
                                             evaluating=True)
                current_distance = run_neat_iteratively(new_network, self.config_path, training=True, sample_size=num_runs)
                print(f"Now at neighbour {i}, with average distance {current_distance}")

                if current_distance > best_distance:  ## aici vreau sa fac 'worst improvement'

                    worst_distance = current_distance
                    best_distance = current_distance
                    print(f"New distance {current_distance}")
                    best_neighbour = neighbour_bitstring

                    found_improvement = True

            if not found_improvement:
                local = False

            try:
                with open(file_tracker, 'a') as f:
                    f.write(f"in iteration {i} the best distance is {worst_distance}\n")
            except IOError:
                print(f"Error when trying to write to file {file_tracker}")

            if best_neighbour is None:
                print("No improvements found, keeping the previous weights and biases")
                best_neighbour = weights_and_biases

            """
            try:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({'network': self.network_path,
                                'num_improvements': i,
                                 'weights_and_biases': best_neighbour,
                                 'best_distance': worst_distance,
                                 'local': local}, f)
            except IOError:
                print(f"Error when writing to pickle {checkpoint_file}")
            """
            # break

        return best_neighbour  # self.network_path


if __name__ == '__main__':

    config_path = "config75"
    network_path = "../models/winner.pkl"

    checkpoint = '../checkpoints/hc_checkpoint.pkl'
    with open(checkpoint, 'rb') as f:
        saved_data = pickle.load(f)
        weights_and_biases = saved_data['weights_and_biases']
        best_distance = saved_data['best_distance']
        print('Checkpoint opened successfully')

    modify_network(network_path, weights_and_biases, -3, 3)


    """
    bitstring = build_continuous_parameters()
    modify_network(network_path, neighbour_bitstring, lower_bound, upper_bound,
                   evaluating=False)
    """

    """
    with open(checkpoint, 'wb') as f:
        pickle.dump({'network': network,
                     'num_improvements': num_improvements,
                     'weights_and_biases': weights_and_biases,
                     'best_distance': best_distance,
                     'local': True}, f)
    
    hill_climbing = HillClimbing(network_path, config_path, precision=5)
    hill_climbing.improve_solution()
    """