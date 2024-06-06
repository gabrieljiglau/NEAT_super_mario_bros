"""
Author: gabriel jiglau
StartingDate: 2024-IV-03
Description: Actual implementation of the NEAT(neuro-evolution of augmenting topologies) algorithm for SuperMarioBros
"""

import numpy as np
import gym_super_mario_bros

# import the Joypad wrapper
from nes_py.wrappers import JoypadSpace

# import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

# import Frame Stacker Wrapper and Gray Scaling Wrapper
from gym.wrappers import FrameStack, GrayScaleObservation

# import VectorizationWrappers
# from stable_baselines3

# import matplotlib
# from matplotlib import pyplot as plt

from NeuralNetwork import Gene, Node, Connection

# global variable used for the initial number of neurons a gene will have
starting_number_of_nodes = 5
threshold = 0.23


class NEAT:

    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.8, mutation_rate_weights: float = 0.04,
                 mutation_rate_connections: float = 0.02, mutation_rate_enable_connections: float = 0.02,
                 mutation_rate_disable_connections: float = 0.01, mutation_rate_nodes: float = 0.03):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate_weights = mutation_rate_weights
        self.number_of_elites = pop_size * 0.085
        self.mutation_rate_connections = mutation_rate_connections
        self.mutation_rate_enable_connections = mutation_rate_enable_connections
        self.mutation_rate_disable_connections = mutation_rate_disable_connections
        self.mutation_rate_nodes = mutation_rate_nodes

        self.current_generation = []

    # needs to be done on connections
    def randomly_initialize_population(self):
        for _ in range(self.pop_size):
            initial_nodes = []
            initial_connections = []

            new_gene = Gene(initial_nodes, initial_connections)
            self.current_generation.append(
                new_gene.initialize_gene_randomly(starting_number_of_nodes, threshold))

    """
        formulate initial population
        randomly initialize population
            repeat
                evaluate objective function
                find fitness function
                apply genetic operators
                    selection
                    crossover
                    mutation
        until stopping criteria
    """

    # am nevoie sa interactionez cu mediul
    # cel mai probabil, voi face o clasa care sa-mi permita acest aspect
    def evaluate_population(self):
        pass

    # smaller values ~0.01 for standard_deviation need to be tested
    def mutate_weights(self, standard_deviation: float = 0.03):
        for individual in self.current_generation:
            individual.mutate_weights_gene(self.mutation_rate_weights, standard_deviation)

    def mutate_enabled_connections(self):
        for individual in self.current_generation:
            individual.mutate_connections_enable_connection(self.mutation_rate_enable_connections)

    def mutate_disabled_connections(self):
        for individual in self.current_generation:
            individual.mutate_connections_disbale_connection(self.mutation_rate_disable_connections)

    def mutate_nodes(self):
        for individual in self.current_generation:
            individual.mutate_nodes_gene(self.mutation_rate_nodes)

    def mutate_connections(self):
        for individual in self.current_generation:
            individual.mutate_connections_gene(self.mutation_rate_connections)

    # TODO: este aceasta varianta corecta? care este diferenta dintre ce este aici si wikipedia?
    def rank_based_selection(self):
        selected_parents = []

        sorted_population = sorted(self.current_generation, key=lambda individual: individual.get_fitness_score())

        population_size = len(sorted_population)
        rank_map = {individual: rank + 1 for rank, individual in enumerate(sorted_population)}

        # Calculate selection probabilities
        selection_probabilities = []
        for individual in sorted_population:
            rank = rank_map[individual]
            probability = 2.0 * (population_size - rank + 1) / (population_size * (population_size + 1))
            selection_probabilities.append(probability)

        # Selection
        for _ in range(2):  # Select two individuals
            random_value = np.random.uniform(0,1)
            cumulative_probability = 0
            selected_individual_index = 0

            for index, probability in enumerate(selection_probabilities):
                cumulative_probability += probability
                if cumulative_probability >= random_value:
                    selected_individual_index = index
                    break

            selected_parents.append(sorted_population[selected_individual_index])

        return tuple(selected_parents)

    '''
    the crossover method takes in 2 compatible genes and produces one offspring 
    '''
    def crossover(self, parent1: Gene, parent2: Gene):

        new_gene = None

        # return at random one of the parents
        if np.random.uniform(0, 1) > self.crossover_rate:
            if np.random.uniform(0, 1) > 0.5:
                return parent1
            else:
                return parent2

        max_innovation_number_first = parent1.previous_innovation_numbers.__len__()
        max_innovation_number_second = parent2.previous_innovation_numbers.__len__()

        bigger_innovation_numbers = {}
        smaller_innovation_numbers = {}
        if max_innovation_number_first > max_innovation_number_second:
            bigger_innovation_numbers = max_innovation_number_first
            smaller_innovation_numbers = max_innovation_number_second
        else:
            bigger_innovation_numbers = max_innovation_number_second
            smaller_innovation_numbers = max_innovation_number_first

        # iterate through the genes using 2 pointers,
        # checking which genes are 'disjoint' and which are 'excess',
        # add them to the new gene


        connections1 = parent1.connections
        connections2 = parent2.connections
        # for i in range(bigger_innovation_numbers.__len__()):
            # mai intai sa verific si daca celalalt are numarul de inovatie
            # apoi sa 'decid' pe baza fitness-ului pe a carui gena o aleg

            # if connections1[i].get_innovation_number.is_connection_enabled


        return new_gene

    def beat_mario(self):
        pass

    def __str__(self):
        return '\n'.join([f"Gene {index}: {individual}" for index, individual in enumerate(self.current_generation)])


if __name__ == '__main__':
    p_size = 10
    cx_rate = 0.8
    m_rate_weights = 0.2
    m_rate_connections = 0.02
    m_rate_nodes = 0.03
    neat_instance = NEAT(p_size, cx_rate, m_rate_weights, m_rate_connections, m_rate_nodes)

    # neat_instance.randomly_initialize_population()

    print("separator")

    # neat_instance.randomly_initialize_population()

    # TODO : test the mutate_nodes and mutate_connections

    node_1 = Node(1)
    node_list = [node_1]
    gene = Gene(node_list)

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    for i in range(5):
        # evaluation_score = gene.evaluate_individual(env, 1000)
        # print(evaluation_score)
        # random_value = np.random.normal(loc=0, scale=0.03)
        zero_to_one = np.random.uniform(0, 1)
        # print(random_value)
        print(zero_to_one)
    # done = True

    initial_position = 0
    # for step in range(5000):
    #     if done:
    #         state = env.reset()
    #     state, reward, done, info = env.step(env.action_space.sample())
    #     print(f"x pos: {info['x_pos']} y pos: {info['y_pos']} time left: {info['time']} world :{info['world']}"
    #           f" status: {info['status']} score: {info['score']}")
    #
    #     current_x_position = info['x_pos']
    #     difference = current_x_position - initial_position
    #     print(f"difference from the starting positions: " + str(difference))
    #     # print(state.world)
    #     env.render()
    #
    # env.close()

    print(neat_instance.__str__())

    """print("hi, mom")

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    done = True

    # loop through each frame in the game
    for frame in range(500):
        if done:
            # start the game
            env.reset()
        # env.action_space.sample() means taking random actions
        state, reward, done, info = env.step(env.action_space.sample())

        # show the game on the screen
        env.render()

    print(env.step(1)[0])
    env.close()"""
