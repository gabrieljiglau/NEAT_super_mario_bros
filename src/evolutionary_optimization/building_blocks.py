import os.path

from src.evolutionary_optimization.utils import get_bit_num, generate_bitstring, decode_discrete, decode_continuous, \
    build_config


def define_parameters(mutate_rate_discrete, mutate_rate_continuous, precision):

    parameters = []

    """
    add each hyperparameter that will be used in NEAT to the final parameters list
    :return: the list that contains all the hyperparameters
    """

    pop_size_values = [i for i in range(5, 100)]
    pop_size = DiscreteParameter(5, 100, mutate_rate_discrete, pop_size_values, precision)
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
    bias_max_value = ContinuousParameter(0, 10, mutate_rate_continuous, precision)
    parameters.append(bias_max_value)
    bias_min_value = ContinuousParameter(-10, 0, mutate_rate_continuous, precision)
    parameters.append(bias_min_value)
    bias_mutate_power = ContinuousParameter(0, 3, mutate_rate_continuous, precision)
    parameters.append(bias_mutate_power)
    bias_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(bias_mutate_rate)
    bias_replace_rate = ContinuousParameter(0, 10, mutate_rate_continuous, precision)
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
    weight_max_value = ContinuousParameter(0, 10, mutate_rate_continuous, precision)
    parameters.append(weight_max_value)
    weight_min_value = ContinuousParameter(-10, 0, mutate_rate_continuous, precision)
    parameters.append(weight_min_value)
    weight_mutate_power = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(weight_mutate_power)
    weight_mutate_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(weight_mutate_rate)
    weight_replace_rate = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(weight_replace_rate)

    compatibility_threshold = DiscreteParameter(1, 5, mutate_rate_discrete, [i for i in range(1, 5)], precision)
    parameters.append(compatibility_threshold)

    max_stagnation = DiscreteParameter(1, 5, mutate_rate_discrete, [i for i in range(1, 5)], precision)
    parameters.append(max_stagnation)
    species_elitism = DiscreteParameter(1, 5, mutate_rate_discrete, [i for i in range(1, 5)], precision)
    parameters.append(species_elitism)

    elitism = DiscreteParameter(1, 5, mutate_rate_discrete, [i for i in range(1, 5)], precision)
    parameters.append(elitism)
    survival_threshold = ContinuousParameter(0, 1, mutate_rate_continuous, precision)
    parameters.append(survival_threshold)

    return parameters


# TODO : i)make sure the generated configs are valid
#       ii)check the validity of the generated configs
#      iii)check why the network has only 3 outputs??
#      iv)build up the genetic algorithm


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
        return generate_bitstring(self.lower_bound, self.upper_bound, self.precision)


class DiscreteParameter(ContinuousParameter):

    def __init__(self, lower_bound: float, upper_bound: float, mutation_rate: float, values: list, precision=5):
        super().__init__(lower_bound, upper_bound, mutation_rate, precision)
        self.values = values
        self.num_values = len(values)
        self.bitstring = ""
        self.bit_length = 0
        self.is_continuous = False

    def initialize_randomly(self):
        return super().initialize_randomly()


class Solution:

    individual_id = 0
    # used to create unique config names;
    # e.g. config1 for the first ever individual, config68 for the 68th individual ever created

    def __init__(self, big_bitstring: str | None, mutate_discrete, mutate_continuous, precision):

        Solution.individual_id += 1

        self.big_bitstring = big_bitstring
        self.decoded_parameters = []
        self.mutate_discrete = mutate_discrete
        self.mutate_continuous = mutate_continuous
        self.precision = precision
        self.fitness = 0
        # the fitness will be the best distance covered by an individual in a run of NEAT
        # with the specified parameters, fixed to a certain generation number (e.g : 50)
        self.parameters = []
        self.config_path = None

    def create_config(self, template):
        config_string = build_config(template, self.decoded_parameters)

        file_path = f"config{self.individual_id}"
        self.config_path = file_path

        try:
            with open(file_path, 'w') as f:
                f.write(config_string)
        except IOError:
            print(f"Error when writing to file {self.config_path}")

    # after each generation, delete the configs for every solution, except the best
    def delete_config(self):
        if self.config_path and os.path.exists(self.config_path):
            os.remove(self.config_path)
            self.config_path = None

    def set_initial_solution(self):
        final_bitstring = []

        self.parameters = define_parameters(self.mutate_discrete, self.mutate_continuous, self.precision)

        for param in self.parameters:
            bitstring = generate_bitstring(param.lower_bound, param.upper_bound, param.precision)

            if not bitstring:
                raise ValueError(f"Generated an empty bitstring for parameter: {param}")

            param.bitstring = bitstring
            final_bitstring.append(bitstring)

        self.big_bitstring = ''.join(final_bitstring)
        print(f"big_bitstring = {self.big_bitstring}")

    def decode_solution(self):

        if len(self.parameters) == 0:
            print("Trying to decode an empty list")
            return

        decoded_parameters = []

        for param in self.parameters:
            if not param.is_continuous:
                param.actual_value = decode_discrete(param.bitstring, param.lower_bound,
                                                     param.upper_bound, param.values)
            elif param.is_continuous:
                param.actual_value = decode_continuous(param.bitstring, param.lower_bound,
                                                       param.upper_bound, param.precision)
            else:
                print("You really shouldn't get here!")
                print("More than 2 categories of parameters detected when decoding the solution!")

            decoded_parameters.append(param.actual_value)

        self.decoded_parameters = decoded_parameters
        return self


if __name__ == '__main__':

    solution = Solution(None, 0.05, 0.04, 5)
    solution.set_initial_solution()

    d_parameters = solution.decode_solution()
    print(f"Decoded_parameters: {d_parameters}")

    template = """
    [NEAT]
    fitness_criterion     = max
    fitness_threshold     = 100000
    pop_size              = {pop_size}
    reset_on_extinction   = {reset_on_extinction}

    [DefaultGenome]
    # node activation options
    activation_default      = {activation_default}
    activation_mutate_rate  = {activation_mutate_rate}
    activation_options      = {activation_options}

    # node aggregation options
    aggregation_default     = {aggregation_default}
    aggregation_mutate_rate = {aggregation_mutate_rate}
    aggregation_options     = sum

    # node bias options
    bias_init_mean          = {bias_init_mean}
    bias_init_stdev         = {bias_init_stdev}
    bias_max_value          = {bias_max_value}
    bias_min_value          = {bias_min_value}
    bias_mutate_power       = {bias_mutate_power}
    bias_mutate_rate        = {bias_mutate_rate}
    bias_replace_rate       = {bias_replace_rate}

    # genome compatibility options
    compatibility_disjoint_coefficient = {compatibility_disjoint_coefficient}
    compatibility_weight_coefficient   = {compatibility_weight_coefficient}

    # connection add/remove rates
    conn_add_prob           = {conn_add_prob}
    conn_delete_prob        = {conn_delete_prob}

    # connection enable options
    enabled_default         = {enabled_default}
    enabled_mutate_rate     = {enabled_mutate_rate}

    feed_forward            =  True
    initial_connection      = {initial_connection}

    # node add/remove rates
    node_add_prob           = {node_add_prob}
    node_delete_prob        = {node_delete_prob}

    # network parameters
    num_hidden              = {num_hidden}
    num_inputs              = 960
    num_outputs             = 3

    # node response options
    response_init_mean      = {response_init_mean}
    response_init_stdev     = {response_init_stdev}
    response_max_value      = {response_max_value}
    response_min_value      = {response_min_value}
    response_mutate_power   = {response_mutate_power}
    response_mutate_rate    = {response_mutate_rate}
    response_replace_rate   = {response_replace_rate}

    # connection weight options
    weight_init_mean        = {weight_init_mean}
    weight_init_stdev       = {weight_init_stdev}
    weight_max_value        = {weight_max_value}
    weight_min_value        = {weight_min_value}
    weight_mutate_power     = {weight_mutate_power}
    weight_mutate_rate      = {weight_mutate_rate}
    weight_replace_rate     = {weight_replace_rate}

    [DefaultSpeciesSet]
    compatibility_threshold = {compatibility_threshold}

    [DefaultStagnation]
    species_fitness_func =  max
    max_stagnation       =  {max_stagnation}
    species_elitism      = {species_elitism}

    [DefaultReproduction]
    elitism            = {elitism}
    survival_threshold = {survival_threshold}
    """

    build_config(template, d_parameters, 'config1')
