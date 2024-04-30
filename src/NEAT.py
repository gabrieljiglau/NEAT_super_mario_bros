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

from BuildingBlocks import Gene, Node, Connection

# global variable used for the initial number of neurons a gene will have
starting_number_of_nodes = 5
threshold = 0.23


class NEAT:

    # oare ar trebui i) o variabila pentru mutatie de adaugat noduri
    #             si ii) o variabila pentur mutatie de adugat conexiuni ?

    def __init__(self, pop_size: int = 40, crossover_rate: float = 0.8, mutation_rate: float = 0.2):
        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.number_of_elites = pop_size * 0.085

        self.current_generation = []

    def randomly_initialize_population(self):
        for i in range(self.pop_size):
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
    def mutate(self, standard_deviation: float = 0.03):
        for individual in self.current_generation:
            individual.mutate_gene(mutation_rate, standard_deviation)

    def tournament_selection(self, population, tournament_size):
        """
        Tournament selection method.

        :param population: List of individuals (genes) to select from.
        :param tournament_size: Number of individuals to compete in each tournament.
        :return: List of selected individuals.
        """
        selected = []

        for _ in range(len(population)):
            # Randomly select tournament_size individuals from the population
            tournament_candidates = np.random.sample(population, tournament_size)

            # Find the best individual (highest fitness) in the tournament
            winner = max(tournament_candidates, key=lambda x: x.evaluate_individual)

            # Add the winner to the selected list
            selected.append(winner)

        return selected

    def beat_mario(self):
        pass

    def __str__(self):
        return '\n'.join([f"Gene {i}: {gene}" for i, gene in enumerate(self.current_generation)])


if __name__ == '__main__':
    p_size = 10
    cx_rate = 0.8
    mutation_rate = 0.2
    neat_instance = NEAT(p_size, cx_rate, mutation_rate)

    # neat_instance.randomly_initialize_population()

    print("separator")

    # neat_instance.randomly_initialize_population()

    node_1 = Node(1)
    node_list = [node_1]
    gene = Gene(node_list)

    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    for i in range(5):
        # evaluation_score = gene.evaluate_individual(env, 1000)
        # print(evaluation_score)
        random_value = np.random.normal(loc=0, scale=0.03)
        print(random_value)
    #done = True

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
