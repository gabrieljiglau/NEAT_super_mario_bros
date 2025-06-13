import math
import numpy as np
import pickle
import random
import tempfile
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

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


def plot_convergence(num_generations, best_distance, q3_distance, average_distance, filename="success_NEAT.png"):
    # Set Seaborn theme for consistent, clean styling
    sns.set_theme(style="whitegrid", font_scale=1.4)

    plt.figure(figsize=(10, 6))

    # Plot the three curves with clear markers and line styles
    plt.plot(
        num_generations, best_distance,
        marker='o', linestyle='-', linewidth=2, markersize=6,
        color='steelblue', label='NEAT', alpha=0.8
    )

    """
    plt.plot(
        num_generations, q3_distance,
        marker='^', linestyle=':', linewidth=2, markersize=6,
        color='darkorange', label='Algoritm hibridizat', alpha=0.8
    )
    """

    """
    plt.plot(
        num_generations, average_distance,
        marker='s', linestyle='dashed', linewidth=2, markersize=6,
        color='mediumseagreen', label='A doua quartilă a distanței' # mediumseagreen
    )
    """

    # Labels and title
    plt.xlabel('Generații', fontsize=14)
    plt.ylabel('Succese', fontsize=14)

    # Grid and ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Legend
    plt.legend(frameon=False)

    # Layout and save
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_chart(time1, time2, time3, time4, filename='runtime_compared.png'):
    sns.set_theme(style='whitegrid', font_scale=1.4)
    plt.figure(figsize=(8, 6))

    # X axis: one for each algorithm
    labels = ['OPT-NEAT', 'HC-NEAT']
    x = np.arange(len(labels))
    bar_width = 0.5

    # Time values: (lumea 2-1-v0, lumea 1-2-v0)
    neat_vals = (time1, time2)
    hc_vals = (time3, time4)

    # Colors: for lumeas
    lumea_labels = ['cu frame-skipping', 'fara frame-skipping']
    lumea_colors = ['steelblue', 'mediumseagreen']

    # Plot each pair: taller bar first, shorter after (overlap)
    for i, (v1, v2) in enumerate([neat_vals, hc_vals]):
        if v1 >= v2:
            plt.bar(x[i], v1, color=lumea_colors[0], width=bar_width, alpha=0.6, label=lumea_labels[0] if i == 0 else "")
            plt.bar(x[i], v2, color=lumea_colors[1], width=bar_width, alpha=1.0, label=lumea_labels[1] if i == 0 else "")
        else:
            plt.bar(x[i], v2, color=lumea_colors[1], width=bar_width, alpha=0.6, label=lumea_labels[1] if i == 0 else "")
            plt.bar(x[i], v1, color=lumea_colors[0], width=bar_width, alpha=1.0, label=lumea_labels[0] if i == 0 else "")

    plt.xticks(x, labels, fontsize=12)
    plt.ylabel('Distanța parcursă', fontsize=12)
    plt.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot(num_iterations, average_distance, filename="distances_hc.png"):
    # Set Seaborn theme for better aesthetics
    sns.set_theme(style="whitegrid", font_scale=1.4)

    plt.figure(figsize=(10, 6))

    # Use a smooth line with markers
    plt.plot(
        num_iterations,
        average_distance,
        marker='o',
        linestyle='-',
        linewidth=2,
        markersize=6,
        color='royalblue',
        label='Distanța medie (eșantion de 10)'
    )

    # Labels and title
    plt.xlabel('Iterații Hill Climbing', fontsize=14)
    plt.ylabel('Distanța Parcursă', fontsize=14)
    plt.title('Evoluția Distanței Medii prin Hill Climbing', fontsize=16)

    # Grid and ticks
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Legend
    plt.legend(frameon=False)

    # Tight layout for saving
    plt.tight_layout()
    plt.savefig(filename, dpi=300)  # High DPI for publication
    plt.close()

def plot_success(improvements, total_generations=5, bin_size=1, filename="winrate_metaGA.png"):
    sns.set_theme(style="whitegrid", font_scale=1.4)

    success_gens = improvements[0]

    # Define bins edges from 0 to total_generations, step bin_size
    bins = np.arange(0, total_generations + bin_size, bin_size)  # e.g. [0,25,50,...,500]

    # Count successes in each bin using numpy histogram
    counts, bin_edges = np.histogram(success_gens, bins=bins)

    # For x axis, take bin centers to plot a smooth line
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.figure(figsize=(10, 6))
    plt.plot(
        bin_centers, counts,
        marker='o', linestyle='-', linewidth=2, markersize=6,
        color='darkorange', label=f'Procentaj de victorii per generație', alpha=0.8
    )

    plt.xlabel('Generații', fontsize=14)
    plt.ylabel('Procentaj victorii', fontsize=14)
    # plt.title(f'Număr succese grupate la fiecare {bin_size} generații', fontsize=16)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(50))
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_winrate(winrates, total_generations=5, bin_size=1, filename="winrate_metaGA.png"):
    sns.set_theme(style="whitegrid", font_scale=1.4)

    # Generate x values (generations or bin centers)
    generations = np.arange(bin_size / 2, total_generations, bin_size)

    plt.figure(figsize=(10, 6))
    plt.plot(
        generations, winrates,
        marker='o', linestyle='-', linewidth=2, markersize=6,
        color='darkorange', label='Procentaj de configurații câștigătoare per generație', alpha=0.8
    )

    plt.xlabel('Generații', fontsize=14)
    # plt.ylabel('Procentaj victorii', fontsize=14)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    plt.ylim(0, 1)  # Limit y-axis from 0 to 1 (0% to 100%)

    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
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


def decode_discrete(bitstring: str, lower: int, upper: int, possible_values: list):
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


def get_hamming_neighbours(current_bitstring, sampling=False):
    """
    :param current_bitstring: the surrounding candidate we search around for an improvement
    :return: a list of size 30, sampled from all neighbours, that differ from the current bitstring in only 1 bit
    """

    bit_list = [int(bit) for bit in current_bitstring]
    neighbours = []
    for i in range(len(current_bitstring)):
        neighbour = bit_list.copy()
        neighbour[i] = 1 - neighbour[i]
        neighbours.append("".join(map(str, neighbour)))

    # print(f"number of neighbours = {len(neighbours)}")
    num_samples = min(30, len(neighbours))
    return random.sample(neighbours, num_samples)


def extract_weights_and_biases(solution_path):
    weights_and_biases = []

    with open(solution_path, 'rb') as winner:
        genome = pickle.load(winner)

    # print(f"genome = \n{genome}\n")

    # print(f"Biases: ")
    for node_key in sorted(genome.nodes.keys()):
        node = genome.nodes[node_key]
        # print(f"Node: {node_key} with bias {node.bias}")
        weights_and_biases.append(node.bias)

    # print(f"Weights: ")
    for conn_key in sorted(genome.connections.keys(), key=lambda x: str(x)):
        conn = genome.connections[conn_key]
        # print(f"Connection {conn_key}: Weight {conn.weight}")
        weights_and_biases.append(conn.weight)

    return weights_and_biases


def get_param_counts(genome):
    return len(genome.nodes), len(genome.connections)

def decode_weights_and_biases(big_bitstring, genome, lower_bound, upper_bound, precision=5):
    step = get_bit_num(lower_bound, upper_bound, precision)
    weights = []
    biases = []

    num_biases, num_connections = get_param_counts(genome)
    num_params = num_biases + num_connections

    if len(big_bitstring) < num_params * step:
        raise ValueError("Bitstring length is too short for this genome")

    for i in range(0, num_params * step, step):
        current_bitstring = big_bitstring[i: i + step]
        decoded_value = decode_continuous(current_bitstring, lower_bound, upper_bound, precision)
        if len(biases) < num_biases:
            biases.append(decoded_value)
        else:
            weights.append(decoded_value)

    return biases, weights


def modify_network(solution_path, encoded_weights_biases, lower_bound, upper_bound, evaluating=False):

    with open(solution_path, 'rb') as winner:
        genome = pickle.load(winner)

    decoded_biases, decoded_weights = decode_weights_and_biases(encoded_weights_biases, genome, lower_bound, upper_bound)

    sorted_conn_keys = sorted(genome.connections.keys(), key=lambda x: str(x))
    sorted_node_keys = sorted(genome.nodes.keys())

    for i, conn_key in enumerate(sorted_conn_keys):
        genome.connections[conn_key].weight = decoded_weights[i]

    for j, node_key in enumerate(sorted_node_keys):
        genome.nodes[node_key].bias = decoded_biases[j]

    if not evaluating:
        with open(solution_path , 'wb') as winner:
            pickle.dump(genome, winner)
        return solution_path, genome

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pkl")
    with open(tmp.name, 'wb') as temp_file:
        pickle.dump(genome, temp_file)
    return tmp.name


if __name__ == '__main__':
    """
    lower_bound = -3
    upper_bound = 3
    precision_p = 5

    decoded_params = extract_weights_and_biases("../models/winner_config75_copy.pkl")
    bitstring = "10100001000110001011001110111001100001110010100101010111000000111000001101001000010010110011110010111000110011101010101100101010111111010111100011001110101010110011110101011010000101100000010111010110100011001110101010111000000010011100010010000000100111000100010101101010101111000101011010101011110010000000100111000100"
    modify_network("../models/winner_config75_copy.pkl", bitstring, lower_bound, upper_bound, False)
    """

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

    max_distance = [
        3161, 3161, 3161, 3161, 1435, 1681, 3161, 1518, 3161, 1435,
        1138, 1654, 1523, 1528, 1441, 1521, 1435, 1427, 1426, 1130
    ]
    q3_distance = [
        2771, 1796, 1795, 1698.0, 1205.5, 1022.0, 2016, 1022.0, 1152, 1008.5,
        922.0, 1122.0, 922.5, 880.25, 1022.0, 922.0, 712.0, 822.0, 1022.0, 822.0
    ]
    average_distance = [
        898.0, 722.0, 722.0, 698.0, 705.5, 722.0, 722.0, 738, 722.0, 708.5,
        722, 722, 722, 680.25, 722.0, 696.00, 643, 733, 820, 722.0
    ]

    num_generations = [i for i in range(10, 100, 10)]
    best_fitness = [
        18129.4846, 18950.3087, 18950.3087, 18950.3087, 18950.3087,
        18950.3087, 18950.3087, 18950.3087, 18950.3087, 18950.3087,
        18950.3087, 18950.3087, 18950.3087, 18950.3087, 18950.3087,
        18950.3087, 18950.3087, 18950.3087, 18950.3087, 18950.3087
    ]

    q3_fitness = [
        4448.6139, 4479.5669, 4469.2493, 4397.9989, 4365.3630,
        4403.0196, 4373.2451, 4306.1123, 4274.7931, 4233.7818,
        4201.2582, 4181.2146, 4143.6527, 4101.4875, 4045.3831,
        3905.3102, 3797.7861, 3749.3820, 3723.1728, 3718.6397
    ]

    average_fitness = [
        1094.8782, 799.5410, 782.1495, 861.0358, 721.9465,
        698.7819, 781.2318, 716.5114, 732.0721, 706.0710,
        724.3559, 814.7628, 794.9985, 826.6955, 737.2056,
        733.5934, 664.2589, 695.9262, 820.4583, 713.8685
    ]

    """
    best_evaluation 5633.4262999999955 in generation 2; mean fitness 3676.107246666666
best_evaluation 7638.738099999982 in generation 3; mean fitness 3627.452469999997
best_evaluation 7638.738099999982 in generation 4; mean fitness 3762.3943633333315
best_evaluation 17556.483499999955 in generation 5; mean fitness 4280.465303333332 
    """

    """
    plot_convergence(num_generations,
                     [1, 1, 1, 2, 0, 1, 1, 0, 0],[0, 1, 0, 2, 0, 0, 0, 0, 0],
                     [723, 822.0, 709, 843, 646, 675, 722.0, 712, 613 ])
    """

    # plot_chart(845, 663, 635, 512 ,'../plots/right_only/frame_skipping1.png')


    improvements = [0.23, 0.43, 0.47, 0.66, 0.71]
    plot_winrate(improvements)
