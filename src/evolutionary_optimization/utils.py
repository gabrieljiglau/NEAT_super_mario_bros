import math
import random


class Randomizer:
    @staticmethod
    def double_between(lower: float, upper: float):
        return lower + (random.uniform(0.0, 1.0) * (upper - lower))

    @staticmethod
    def get_int_between(lower: int, upper: int):
        return random.randint(lower, upper)

    @staticmethod
    def next_double():
        return random.uniform(0.0, 1.0)


def generate_bitstring(lower: float, upper: float, precision: int):
    bit_list = []
    size = get_bit_num(lower, upper, precision)

    for i in range(size):
        p1 = Randomizer.next_double()
        p2 = Randomizer.next_double()

        if p1 > p2:
            bit_list.append('1')
        else:
            bit_list.append('0')

    return ''.join(bit_list)

def decode_continuous(bitstring: str, lower: float, upper: float, precision: int):

    value = int(bitstring, 2)
    target = upper - lower
    max_value = (1 << len(bitstring)) - 1

    decoded_value = lower + (value / max_value) * target
    return round(decoded_value, precision)


def add_possible_mutation(input_bistring: str, mutation_rate: float):

    new_bitstring = []

    for bit in input_bistring:

        generated_num = Randomizer.next_double()

        if generated_num < mutation_rate:
            new_bit = '1' if bit == '0' else '1'
        else:
            new_bit = bit

        new_bitstring.append(new_bit)

    return ''.join(new_bitstring)

def encode_continuous():
    pass


def get_bit_num(lower: float, upper: float, precision: int = 2):
    sub_intervals = (upper - lower) * math.pow(10, precision)

    log_n = math.log2(sub_intervals)
    log_2 = math.log2(2)

    return int(math.ceil(log_n / log_2))


def plot_learning_curve(x, scores, figure_file):
    pass


if __name__ == '__main__':

    lower_bound = 0
    upper_bound = 10
    precision_p = 5

    generated_bitstring = generate_bitstring(lower_bound, upper_bound, precision_p)

    mutated_bistring = add_possible_mutation(generated_bitstring, 0.05)

    original_num = decode_continuous(generated_bitstring, lower_bound, upper_bound, precision_p)
    mutated_num = decode_continuous(mutated_bistring, lower_bound, upper_bound, precision_p)

    print(f"Generated bitstring: {generated_bitstring} of the value {original_num}")
    print(f"Mutated bitstring: {mutated_bistring} of the value {mutated_num}")
