from enum import Enum
from src.evolutionary_optimization.utils import get_bit_num


# TODO: i)continue with the other classes for the Discrete and BooleanParameter
#       ii) combine all continuous, discrete and boolean parameters in a list of their own
class ContinuousParameter:

    def __init__(self, lower_bound: float, upper_bound: float, bit_string: str, mutation_rate: float, precision=3):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.bit_string = bit_string
        self.mutation_rate = mutation_rate
        self.precision = precision
        self.is_continuous = True
        self.is_boolean = False
        self.is_discrete = False


"""
class DiscreteParameter(ContinuousParameter):

    def __init__(self, lower_bound: float, upper_bound: float, bit_string: str,
                 mutation_rate: float, is_boolean, precision=3):
        if is_boolean:

        super.__init__(lower_bound,)
"""


class HyperparameterSize(Enum):
    POP_SIZE = 17
    RESET_ON_EXTINCTION = 1  # True/False

    # when converting, take Math.ceil(converted_digit) if converted_digit 'closer to ceil', otherwise math.upper
    ACTIVATION_DEFAULT = 13
    ACTIVATION_MUTATE_RATE = 10
    ACTIVATION_OPTIONS = 13
    AGGREGATION_DEFAULT = 6
    AGGREGATION_MUTATE_RATE = 10

    BIAS_INIT_MEAN = 11  # in [-1, 1]
    BIAS_INIT_STDDEV = 11  # in [-1, 1]
    BIAS_MAX_VALUE = 14
    BIAS_MIN_VALUE = 14
    BIAS_MUTATE_POWER = 13
    BIAS_MUTATE_RATE = 14
    BIAS_REPLACE_RATE = 9  # in [0, 0.5]

    COMPATIBILITY_DISJOINT_COEFFICIENT = 13
    COMPATIBILITY_WEIGHT_COEFFICIENT = 13



# in total, there are 43 hyperparameters
class Solution:

    def __init__(self, bitstring: str, mutate_pop_size, mutate_boolean, mutate_discrete, mutate_continuous):
        self.bitstring = bitstring
        self.mutate_pop_size = mutate_pop_size
        self.mutate_boolean = mutate_boolean
        self.mutate_discrete = mutate_discrete
        self.mutate_continuous = mutate_continuous
        self.fitness = 0

    def set_initial_solution(self):
        pass


if __name__ == '__main__':
    print(get_bit_num(0, 5, 3))
