"""
Author: gabriel jiglau
StartingDate: 2024-IV-03
Description: Actual implementation of the NEAT(neuro-evolution of augmenting topologies) algorithm for SuperMarioBros
"""

# import the game
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

import numpy as np

import random
from BuildingBlocks import Gene, Node, Connection, NodesNotConnectedException

# global variable used for the initial number of neurons a gene will have
starting_number_of_nodes = 5
threshold = 0.23


class NEAT:

    # oare ar trebui i) o variabila pentru mutatie de adaugat noduri
    #             si ii) o variabila pentur mutatie de adugat conexiuni ?

    def __init__(self, pop_size=40, crossover_rate=0.8, mutation_rate=0.2):
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
            self.current_generation.append(new_gene.initialize_gene_randomly(starting_number_of_nodes, threshold))

    """
        formulate initial population
        randomly initialize population
            repeat
                evaluate objective function
                find fitness function
                apply genetic operators
                    reproduction
                    crossover
                    mutation
        until stopping criteria
    """

    # am nevoie sa interactionez cu mediul
    # cel mai probabil, voi face o clasa care sa-mi permita acest aspect
    def evaluate_population(self):
        pass

    def beat_mario(self):
        pass

    def __str__(self):
        return '\n'.join([f"Gene {i}: {gene}" for i, gene in enumerate(self.current_generation)])


if __name__ == '__main__':
    pop_size = 20
    crossover_rate = 0.8
    mutation_rate = 0.2
    neat_instance = NEAT(pop_size, crossover_rate, mutation_rate)

    neat_instance.randomly_initialize_population()

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
