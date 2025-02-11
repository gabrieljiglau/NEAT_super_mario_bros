import os.path
from typing import List
from evolutionary_optimization.utils import generate_bitstring, decode_discrete, decode_continuous, \
    build_config, import_template, add_possible_mutation, get_bit_num

def define_and_initialize_parameters(mutate_rate_discrete, mutate_rate_continuous, precision):
    parameters = []

    """
    add each hyperparameter that will be used in NEAT to the final parameters list
    :return: the list containing all the encoded hyperparameters
    """

    pop_size_values = [i for i in range(10, 20)]
    pop_size = DiscreteParameter(10, 20, mutate_rate_discrete, pop_size_values, precision)
    parameters.append(pop_size)

    boolean_values = [False, True]
    reset_on_extinction = DiscreteParameter(0, 1, mutate_rate_discrete, boolean_values, precision)
    parameters.append(reset_on_extinction)

    activation_values = ["tanh", "square", "gauss", "sigmoid", "relu", "log", "clamped"]
    activation_default = DiscreteParameter(0, 6, mutate_rate_discrete, activation_values, precision)
    parameters.append(activation_default)

    activation_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(activation_mutate_rate)

    activation_options = DiscreteParameter(0, 6, mutate_rate_discrete, activation_values, precision)
    parameters.append(activation_options)

    aggregation_values = ["min", "max", "sum", "product", "median", "mean"]
    aggregation_default = DiscreteParameter(0, 5, mutate_rate_discrete, aggregation_values, precision)
    parameters.append(aggregation_default)

    aggregation_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(aggregation_mutate_rate)

    bias_init_mean = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(bias_init_mean)

    bias_init_stddev = ContinuousParameter(0, 3, mutate_rate_continuous, precision)
    parameters.append(bias_init_stddev)

    bias_max_value = ContinuousParameter(0, 3, mutate_rate_continuous, precision)
    parameters.append(bias_max_value)

    bias_min_value = ContinuousParameter(-3, 0, mutate_rate_continuous, precision)
    parameters.append(bias_min_value)

    bias_mutate_power = ContinuousParameter(0, 3, mutate_rate_continuous, precision)
    parameters.append(bias_mutate_power)

    bias_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(bias_mutate_rate)

    bias_replace_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(bias_replace_rate)

    compatibility_disjoint_coefficient = ContinuousParameter(0, 5, mutate_rate_continuous, precision)
    parameters.append(compatibility_disjoint_coefficient)

    compatibility_weight_coefficient = ContinuousParameter(0, 5, mutate_rate_continuous, precision)
    parameters.append(compatibility_weight_coefficient)

    conn_add_prob = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(conn_add_prob)

    conn_delete_prob = ContinuousParameter(0, 1, mutate_rate_continuous, 1)
    parameters.append(conn_delete_prob)

    enabled_default = DiscreteParameter(0, 1, mutate_rate_discrete, boolean_values, precision)
    parameters.append(enabled_default)

    enabled_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(enabled_mutate_rate)

    connection_properties = ["unconnected", "fs_neat_nohidden", "fs_neat_hidden", "full_nodirect", "full_direct"]
    initial_connection = DiscreteParameter(0, 4, mutate_rate_discrete, connection_properties, precision)
    parameters.append(initial_connection)

    node_add_prob = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(node_add_prob)

    node_delete_prob = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(node_delete_prob)

    hidden_size = [i for i in range(0, 10)]
    num_hidden = DiscreteParameter(0, 10, mutate_rate_discrete, hidden_size, precision)
    parameters.append(num_hidden)

    response_init_mean = ContinuousParameter(0, 5, mutate_rate_continuous, precision)
    parameters.append(response_init_mean)

    response_init_stddev = ContinuousParameter(0, 5, mutate_rate_continuous, precision)
    parameters.append(response_init_stddev)

    response_max_value = ContinuousParameter(0, 30, mutate_rate_continuous, precision)
    parameters.append(response_max_value)

    response_min_value = ContinuousParameter(-30, 0, mutate_rate_continuous, precision)
    parameters.append(response_min_value)

    response_mutate_power = ContinuousParameter(0, 3, mutate_rate_continuous, precision)
    parameters.append(response_mutate_power)

    response_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(response_mutate_rate)

    response_replace_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(response_replace_rate)

    weight_init_mean = ContinuousParameter(-1, 1, mutate_rate_continuous, precision)
    parameters.append(weight_init_mean)

    weight_init_stddev = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(weight_init_stddev)

    weight_max_value = ContinuousParameter(0, 3, mutate_rate_continuous, precision)
    parameters.append(weight_max_value)

    weight_min_value = ContinuousParameter(-3, 0, mutate_rate_continuous, precision)
    parameters.append(weight_min_value)

    weight_mutate_power = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(weight_mutate_power)

    weight_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(weight_mutate_rate)

    weight_replace_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(weight_replace_rate)

    compatibility_threshold = DiscreteParameter(10, 20, mutate_rate_discrete, [i for i in range(10, 20)], precision)
    parameters.append(compatibility_threshold)

    max_stagnation = DiscreteParameter(1, 15, mutate_rate_discrete, [i for i in range(1, 15)], precision)
    parameters.append(max_stagnation)

    species_elitism = DiscreteParameter(1, 5, mutate_rate_discrete, [i for i in range(1, 5)], precision)
    parameters.append(species_elitism)

    elitism = DiscreteParameter(1, 5, mutate_rate_discrete, [i for i in range(1, 5)], precision)
    parameters.append(elitism)

    survival_threshold = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(survival_threshold)

    final_bitstring = []
    for param in parameters:
        bitstring = param.initialize_randomly()

        if not bitstring:
            raise ValueError(f"Generated an empty bitstring for parameter: {param}")

        final_bitstring.append(bitstring)

    return parameters


"""
decoded parameters should be returned after the initialization of the encoded parameters has taken place
"""
def decode_parameters(encoded_parameters):
    decoded_parameters = []
    for param in encoded_parameters:
        if param.is_continuous:
            decoded_parameters.append(param.decode_continuous())
        else:
            decoded_parameters.append(param.decode_discrete())

    return decoded_parameters


class ContinuousParameter:

    def __init__(self, lower_bound: float, upper_bound: float, mutation_rate: float, precision=5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bitstring = ""
        self.bit_length = 0
        self.mutation_rate = mutation_rate
        self.precision = precision
        self.actual_value = None
        self.is_continuous = True

    def initialize_randomly(self):
        self.bitstring = generate_bitstring(self.lower_bound, self.upper_bound, self.precision)
        self.bit_length = len(self.bitstring)
        return self.bitstring

    def decode_continuous(self):
        self.actual_value = decode_continuous(self.bitstring, self.lower_bound, self.upper_bound, self.precision)
        return self.actual_value


class DiscreteParameter(ContinuousParameter):

    def __init__(self, lower_bound: int, upper_bound: int, mutation_rate: float, values: list, precision=5):
        super().__init__(lower_bound, upper_bound, mutation_rate, precision)
        self.values = values
        self.num_values = len(values)
        self.bitstring = ""
        self.bit_length = 0
        self.is_continuous = False

    def initialize_randomly(self):
        return super().initialize_randomly()

    def decode_continuous(self):
        raise AttributeError("decode_continuous is not available for DiscreteParameter")

    def decode_discrete(self):
        self.actual_value = decode_discrete(self.bitstring, int(self.lower_bound), int(self.upper_bound), self.values)
        return self.actual_value


class IndividualSolution:
    """
    the class holds the appropriate parameters and methods needed for one single individual
    """

    individual_id = 0
    """
    used to create unique config names;
    e.g. config1 for the first ever individual, config68 for the 68th individual ever created etc.
    """

    def __init__(self, big_bitstring=None, config_path=None, mutate_continuous=None, mutate_discrete=None,
                 precision=5):

        IndividualSolution.individual_id += 1

        if big_bitstring is None:
            self.big_bitstring = ""
        else:
            self.big_bitstring = big_bitstring

        self.parameters = define_and_initialize_parameters(mutate_discrete, mutate_continuous, precision)
        self.decoded_parameters = decode_parameters(self.parameters)
        """
        there should be an explicit decoding of the parameters -> after crossover
        """

        if config_path is None:
            self.config_path = self.create_config()
        else:
            self.config_path = config_path
        self.fitness = 0
        """
        the fitness will be the best distance covered by an individual in a complete fixed execution of NEAT
        with the specified parameters, fixed to a certain generation number (e.g : 50)
        """

    def __getitem__(self, index):
        return self.parameters[index]

    def create_config(self):

        template = import_template()

        current_dir = os.path.dirname(__file__)
        parent_dir = os.path.dirname(current_dir)
        base_dir = os.path.join(parent_dir, 'configs')
        os.makedirs(base_dir, exist_ok=True)
        file_name = f"config{self.individual_id}"
        file_path = os.path.join(base_dir, file_name)
        self.config_path = file_path

        config_string = build_config(template, self.decoded_parameters, self.config_path)

        # print(f"config_string = {config_string}")

        try:
            with open(file_path, 'w') as f:
                f.write(config_string)
            return file_path
        except IOError:
            print(f"Error when writing to file {self.config_path}")

    def delete_config(self):
        """
        after each generation, you should delete the config created files for every solution, except the best
        """

        if self.config_path and os.path.exists(self.config_path):
            os.remove(self.config_path)
            self.config_path = None

    def set_initial_solution(self, mutate_discrete: float, mutate_continuous: float, precision: int):

        final_bitstring = []
        self.parameters = define_and_initialize_parameters(mutate_discrete, mutate_continuous, precision)

        for param in self.parameters:
            bitstring = param.bitstring

            if not bitstring:
                raise ValueError(f"Empty bitstring found for parameter: {param}")

            final_bitstring.append(bitstring)

        self.decoded_parameters = decode_parameters(encoded_parameters=self.parameters)
        self.big_bitstring = ''.join(final_bitstring)

    @classmethod
    def mutate_individual_solution(cls, individual, mutate_discrete: float, mutate_continuous: float) -> str:
        """
        Class method to mutate an individual solution.

        Args:
            cls: The class itself (passed automatically for class methods).
            individual: The individual solution to mutate (an instance of IndividualSolution).
            mutate_discrete: Probability for discrete mutation.
            mutate_continuous: Probability for continuous mutation.

        Returns:
            The mutated bitstring of the individual solution.
        """

        result = []
        current_index = 0

        for param in individual.parameters:
            bit_count = get_bit_num(param.lower_bound, param.upper_bound, param.precision)
            current_bitstring = individual.big_bitstring[current_index:current_index + bit_count]

            if param.is_continuous:
                result.append(add_possible_mutation(current_bitstring, mutate_continuous))
            else:
                result.append(add_possible_mutation(current_bitstring, mutate_discrete))

            current_index += bit_count

        individual.big_bitstring = ''.join(result)
        print(f"Bitstring after mutation: {individual.big_bitstring}")

        return individual.big_bitstring

    def decode_individual_solution(self):
        """
        iterates over all the parameters, keeping track of the gene's position and decoding it, based on the
        nature of the parameter (discrete or continuous)
        """

        if len(self.parameters) == 0:
            print("Trying to decode an empty list")
            return

        decoded_parameters = []
        current_index = 0

        for param in self.parameters:

            bit_count = get_bit_num(param.lower_bound, param.upper_bound, param.precision)
            param_bitstring = self.big_bitstring[current_index:current_index + bit_count]

            if not param.is_continuous:
                param.actual_value = decode_discrete(param_bitstring, int(param.lower_bound),
                                                     int(param.upper_bound), param.values)
            elif param.is_continuous:
                param.actual_value = decode_continuous(param_bitstring, param.lower_bound,
                                                       param.upper_bound, param.precision)
            else:
                print("Unexpected parameter type detected during decoding!")

            decoded_parameters.append(param.actual_value)
            current_index += bit_count

        self.decoded_parameters = decoded_parameters
        return self.decoded_parameters


class Solution:
    """
    the solution class aggregates a whole generation, with each individual being of type IndividualSolution
    """

    def __init__(self, solution_list: List[IndividualSolution] = None):
        if solution_list is None:
            self.solution_list = []
        else:
            self.solution_list = solution_list

    @classmethod
    def create_first_generation(cls, pop_size, mutate_discrete, mutate_continuous, precision):

        new_generation = []
        for _ in range(pop_size):
            individual = IndividualSolution()
            individual.set_initial_solution(mutate_discrete, mutate_continuous, precision)
            new_generation.append(individual)

        return cls(new_generation)


if __name__ == '__main__':
    solution = IndividualSolution()
    solution.set_initial_solution(0.08, 0.07, 5)
    print(f"Decoded solution: {solution.decode_individual_solution()}")

    IndividualSolution.mutate_individual_solution(solution, 0.42, 0.57)
    print(f"Decoded solution: {solution.decode_individual_solution()}")
