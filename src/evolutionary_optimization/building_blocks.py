from src.evolutionary_optimization.utils import get_bit_num, generate_bitstring


def define_parameters(mutate_rate_discrete, mutate_rate_continuous, precision):

    parameters = []

    """
    add each hyperparameter that will be used in NEAT to the final parameters list
    :return: the list that contains all the hyperparameters
    """

    pop_size_values = [i for i in range(5, 100)]
    print(pop_size_values)
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

    # TODO: i)add all hyperparameters to the list
    #      ii)initialize and decode properly the big bitstring from the Solution class
    #     iii)make sure the generated configs are valid

    BIAS_INIT_MEAN = 11  # in [-1, 1]
    BIAS_INIT_STDDEV = 11  # in [-1, 1]
    BIAS_MAX_VALUE = 14
    BIAS_MIN_VALUE = 14
    BIAS_MUTATE_POWER = 13
    BIAS_MUTATE_RATE = 14
    BIAS_REPLACE_RATE = 9  # in [0, 0.5]

    COMPATIBILITY_DISJOINT_COEFFICIENT = 13
    COMPATIBILITY_WEIGHT_COEFFICIENT = 13

class ContinuousParameter:

    def __init__(self, lower_bound: float, upper_bound: float, mutation_rate: float, precision=5):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bit_string = ""
        self.bit_length = 0
        self.mutation_rate = mutation_rate
        self.precision = precision
        self.is_continuous = True

    def initialize_randomly(self):
        return generate_bitstring(self.lower_bound, self.upper_bound, self.precision)

class DiscreteParameter(ContinuousParameter):

    def __init__(self, lower_bound:float, upper_bound: float, mutation_rate: float, values: list, precision=5):
        super().__init__(lower_bound, upper_bound, mutation_rate, precision)
        self.values = values
        self.num_values = len(values)
        self.bit_string = ""
        self.bit_length = 0
        self.is_continuous = False

    def initialize_randomly(self):
        return super().initialize_randomly()

class Solution:

    def __init__(self, big_bitstring: str, mutate_discrete, mutate_continuous):
        self.big_bitstring = big_bitstring
        self.mutate_discrete = mutate_discrete
        self.mutate_continuous = mutate_continuous
        self.fitness = 0

    # the fitness will be the best distance covered by an individual in a run of NEAT
    # with the specified parameters, fixed to a certain generation number (e.g : 50)

    def set_initial_solution(self):
        pass


if __name__ == '__main__':
    print(get_bit_num(0, 5, 3))
