import math
import pickle
import random
import matplotlib.pyplot as plt

base_template = """
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
num_outputs             = 5

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

def import_template():
    return base_template

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

def plot_convergence(num_individuals, num_species):
    plt.figure(figsize=(10, 6))

    generations = range(len(num_individuals))

    plt.plot(generations, num_individuals, marker='o', linestyle='-', label='Number of individuals in a generation')
    plt.plot(generations, num_species, marker='s', linestyle='dashed', label='Number of species')

    plt.xlabel('Generations')
    plt.ylabel('Individuals')
    plt.title('Number of individuals ans species per generations')
    plt.legend()
    plt.savefig('individuals_and_species_across_generations.png')
    plt.close()

def generate_bitstring(lower: float, upper: float, precision: int):

    """
    this function can be called for generating valid bit-strings for both discrete and continuous parameters
    :returns: a bitstring that's always inside [lower, upper]
    """

    if lower >= upper:
        raise ValueError(f"Invalid bounds: lower_bound={lower}, upper_bound={upper}")

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

def encode_continuous(input_num: float, lower: float, upper: float, precision: int):

    if not (lower <= input_num <= upper):
        raise ValueError(f"Input number {input_num} is out of bounds [{lower}, {upper}]")

    bit_size = get_bit_num(lower, upper, precision)

    # scale the input number to an integer range [0, max_value]
    max_value = (1 << bit_size) - 1  # 2^bit_size - 1
    scaled_value = round(((input_num - lower) / (upper - lower)) * max_value)

    bitstring = format(scaled_value, f'0{bit_size}b')

    return bitstring

def decode_discrete(bitstring: str, lower: int, upper:int, possible_values: list):

    """
    return the item at a given index from the input list
    """

    value = int(bitstring, 2)
    target = upper - lower
    max_value = (1 << len(bitstring)) - 1

    scaled_value = lower + (value / max_value) * target

    floor = int(scaled_value)
    ceiling = floor + 1

    floor = max(0, min(floor, len(possible_values) - 1))
    ceiling = max(0, min(ceiling, len(possible_values) - 1))

    # choose the closer value between possible_values[floor] and possible_values[ceiling]
    if abs(scaled_value - floor) <= abs(scaled_value - ceiling):
        return possible_values[floor]
    else:
        return possible_values[ceiling]

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

def get_bit_num(lower: float, upper: float, precision: int = 5):
    sub_intervals = (upper - lower) * math.pow(10, precision)

    log_n = math.log2(sub_intervals)
    log_2 = math.log2(2)

    return int(math.ceil(log_n / log_2))

def build_config(template, decoded_parameters, filename):

    """
    Builds a configuration file from a template and a list of decoded parameters.
    :param decoded_parameters: List of decoded parameters to replace placeholders.
    :param template: (str) The configuration template containing placeholders.
    :param filename: Path to save the newly generated filename

    Returns:
        str: The completed configuration file as a string.
    """

    params_map = {
        "pop_size": decoded_parameters[0],
        "reset_on_extinction": decoded_parameters[1],
        "activation_default": decoded_parameters[2],
        "activation_mutate_rate": decoded_parameters[3],
        "activation_options": decoded_parameters[4],
        "aggregation_default": decoded_parameters[5],
        "aggregation_mutate_rate": decoded_parameters[6],
        "bias_init_mean": decoded_parameters[7],
        "bias_init_stdev": decoded_parameters[8],
        "bias_max_value": decoded_parameters[9],
        "bias_min_value": decoded_parameters[10],
        "bias_mutate_power": decoded_parameters[11],
        "bias_mutate_rate": decoded_parameters[12],
        "bias_replace_rate": decoded_parameters[13],
        "compatibility_disjoint_coefficient": decoded_parameters[14],
        "compatibility_weight_coefficient": decoded_parameters[15],
        "conn_add_prob": decoded_parameters[16],
        "conn_delete_prob": decoded_parameters[17],
        "enabled_default": decoded_parameters[18],
        "enabled_mutate_rate": decoded_parameters[19],
        "initial_connection": decoded_parameters[20],
        "node_add_prob": decoded_parameters[21],
        "node_delete_prob": decoded_parameters[22],
        "num_hidden": decoded_parameters[23],
        "response_init_mean": decoded_parameters[24],
        "response_init_stdev": decoded_parameters[25],
        "response_max_value": decoded_parameters[26],
        "response_min_value": decoded_parameters[27],
        "response_mutate_power": decoded_parameters[28],
        "response_mutate_rate": decoded_parameters[29],
        "response_replace_rate": decoded_parameters[30],
        "weight_init_mean": decoded_parameters[31],
        "weight_init_stdev": decoded_parameters[32],
        "weight_max_value": decoded_parameters[33],
        "weight_min_value": decoded_parameters[34],
        "weight_mutate_power": decoded_parameters[35],
        "weight_mutate_rate": decoded_parameters[36],
        "weight_replace_rate": decoded_parameters[37],
        "compatibility_threshold": decoded_parameters[38],
        "max_stagnation": decoded_parameters[39],
        "species_elitism": decoded_parameters[40],
        "elitism": decoded_parameters[41],
        "survival_threshold": decoded_parameters[42],
    }

    try:
        formatted_config = template.format(**params_map)
        print(f"Formatted configuration: {formatted_config}")  # Debugging output
    except KeyError as e:
        print(f"KeyError: Missing key in params_map - {e}")
        return None

    try:
        with open(filename, "w") as f:
            f.write(formatted_config)
        # print(f"Configuration written to {filename}")
    except IOError as e:
        print(f"IOError occurred when writing formatted config to {filename}: {e}")

    return formatted_config

def build_continuous_parameters(input_values, lower_bound, upper_bound):

    parameters = []
    for i in range(len(input_values)):

        encoded_bitstring = encode_continuous(input_values[i], lower_bound, upper_bound, precision=5)
        parameters.append(encoded_bitstring)

    return ''.join(parameters)

def get_hamming_neighbours(current_bitstring):

    """
    :param current_bitstring: the surrounding candidate we search around for an improvement
    :return: a list of size 30, sampled from all neighbours, that differ from the current bitstring in only 1 bit
    """

    bit_list = [int(bit) for bit in current_bitstring]
    neighbours = []
    for i in range(len(current_bitstring)):
        neighbour = bit_list.copy()
        neighbour[i] = 1 - neighbour[i]
        neighbours.append("".join(map(str,neighbour)))

    # print(f"number of neighbours = {len(neighbours)}")
    num_samples = min(30, len(neighbours))
    return random.sample(neighbours, num_samples)

def extract_weights_and_biases(solution_path):

    weights_and_biases = []

    with open(solution_path, 'rb') as winner:
        genome = pickle.load(winner)

    print(f"genome = \n{genome}\n")
    full_weights = {key: conn.weight for key, conn in genome.connections.items()}

    """
    !!! HERE I WANT THE WEIGHTS AND BIASES TO BE EXTRACTED IN THE EXACT SAME ORDER THAY ARE STORED IN PICKLE !!
    """

    print(f"Weights: ")
    for conn, weight in full_weights.items():
        print(f"Connection {conn}: Weight {weight}")
        weights_and_biases.append(weight)

    print(f"Biases: ")
    full_biases = {key: node.bias for key, node in genome.nodes.items()}
    for node, bias in full_biases.items():
        print(f"Node: {node} with bias {bias}")
        weights_and_biases.append(bias)


    return weights_and_biases

def modify_network(solution_path, encoded_weights_biases, lower_bound, upper_bound, evaluating_candidates=False):

    with open(solution_path, 'rb') as winner:
        genome = pickle.load(winner)

    decoded_biases, decoded_weights = decode_weights_and_biases(encoded_weights_biases, lower_bound, upper_bound)

    """
    problema aici, fiindca nu pare ca lista de ponderi si bias-uri este in concordanță cu ceea ce se află în pickle
    """

    print(f"biases = {decoded_biases}")
    print(f"weights = {decoded_weights}")

    """
    weights_and_biases = []
    print(f"Weights: ")
    for conn, weight in full_weights.items():
        print(f"Connection {conn}: Weight {weight}")
        weights_and_biases.append(weight)

    print(f"Biases: ")
    full_biases = {key: node.bias for key, node in genome.nodes.items()}
    for node, bias in full_biases.items():
        print(f"Node: {node} with bias {bias}")

    """

    for i, conn_key in enumerate(genome.connections.keys()):
        genome.connections[conn_key].weight = decoded_weights[i]


    for j, node_key in enumerate(genome.nodes.keys()):
        genome.nodes[node_key].bias = decoded_biases[j]

    if not evaluating_candidates:
        with open(solution_path, 'wb') as winner:
            pickle.dump(genome, winner)
            print('Updated genome successfully')


def decode_weights_and_biases(big_bitstring, lower_bound, upper_bound, num_biases=11, num_params=16):

    step = get_bit_num(lower_bound, upper_bound, precision=5)
    weights = []
    biases = []

    if len(big_bitstring) < num_params *  step:
        raise ValueError("Bitstring length is too short")

    for i in range(0, num_params * step, step):
        current_bitstring = big_bitstring[i: i + step]
        decoded_value = decode_continuous(current_bitstring, lower_bound, upper_bound, precision=5)
        if len(biases) < num_biases :
            biases.append(decoded_value)
        else:
            weights.append(decoded_value)

    return biases, weights


if __name__ == '__main__':


    lower_bound = -3
    upper_bound = 3
    precision_p = 5

    decoded_params = extract_weights_and_biases("../models/winner_config75_copy.pkl")
    bitstring = build_continuous_parameters(decoded_params, lower_bound, upper_bound)
    modify_network("../models/winner_config75_copy.pkl", bitstring, lower_bound, upper_bound, False)



    """
    Connection(-52, 627): Weight - 0.96865
    Connection(627, 422): Weight
    0.01431
    Connection(-204, 1): Weight
    0.01431
    Connection(-515, 627): Weight
    0.01431
    """

    # print(get_bit_num(lower_bound, upper_bound, precision_p))
    """
    float_list = [-0.96865, 0.01431, 0.01431, 0.01431]
    arr = []

    for i in range(len(float_list)):
        arr.append(encode_continuous(float_list[i],lower_bound, upper_bound, precision_p))

    print(arr)
    """
    print(get_hamming_neighbours("01010110101010111100100000001001110001001000000010011100010010000000100111000100"))

    """
    generated_bitstring = generate_bitstring(lower_bound, upper_bound, precision_p)
    decoded_bitstring = decode_continuous(generated_bitstring, lower_bound, upper_bound, precision_p)
    print(f"generated_bitstring = {generated_bitstring}")
    print(f"decoded_bitstring = {decoded_bitstring}")
    print(f"encoded_bitstring = {encode_continuous(decoded_bitstring, lower_bound, upper_bound, precision_p)}")
    # mutated_bistring = add_possible_mutation(generated_bitstring, 0.15)
    """

    """
    values = ["ala", "bala", "portocala", "cine", "mi-a", "mancat", "banana"]
    original_num = decode_discrete(generated_bitstring, lower_bound, upper_bound, values)
    # mutated_num = decode_discrete(mutated_bistring, lower_bound, upper_bound, values)

    print(f"Generated bitstring: {generated_bitstring} of the value {original_num}")
    """

    """
    fitness_list = [2502.849, 3869.256, 2482.458, 2561.222, 3701.758, 6606.893, 17866.112]
    average_fitness = [791.359, 898.897, 818.48431, 832.592, 830.129, 822.492, 855.986]
    generation_time = [45.219, 62.222, 128.921, 291.252, 483.491, 653.161, 41185.393]
    plot_convergence(fitness_list, average_fitness, generation_time)
    """
    """
    no_individuals = [79, 190, 396, 660, 897, 1392, 1948]
    no_species = [47, 99, 165, 241, 348, 487, 646]
    plot_convergence(no_individuals, no_species)
    """