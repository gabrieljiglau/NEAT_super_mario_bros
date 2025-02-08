import os
import cv2
import neat
import pickle
import numpy as np
import multiprocessing
import warnings; warnings.warn = lambda *args,**kwargs: None
import gym_super_mario_bros
from collections import deque
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

"""
This code is based on:
Source: https://github.com/asarathy28/smb_neat/blob/main/SMB/simple_movement/simple_SMB.py
Author: asarathy28
License: (Check the repository for the applicable license)
"""

STACK_SIZE = 4
frame_stack = deque(maxlen=STACK_SIZE)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

generation = 0

best_individual_fitness = float('-inf')
best_individual_stats = {"distance": 0, "time": 0, "fitness": 0}

success_counter = multiprocessing.Value("i", 0)

def stack_frames(stack, new_frame, is_new_episode):

    if is_new_episode:
        stack.clear()
        for _ in range(STACK_SIZE):
            stack.append(new_frame)
    else:
        stack.append(new_frame)

    return np.stack(stack, axis=0)

def eval_genome(genome, config):
    """Evaluates a single genome in NEAT."""
    global generation, best_individual_fitness, best_individual_stats

    observation = env.reset()
    neural_network = neat.nn.FeedForwardNetwork.create(genome, config)

    width, height, color = env.observation_space.shape
    width = int(width / 8)
    height = int(height / 8)

    best_fitness = 0
    genome.fitness = 0
    frame = 0
    counter = 0
    xpos_max = 0
    distance_traveled = 0
    done = False
    success = False  # Track if this genome succeeds

    while not done:

        observation = preprocess(observation, width, height)
        observation = observation / 255.0  # normalize to range [0,1]
        stacked_observation = stack_frames(frame_stack, observation, is_new_episode=False)
        image_array = stacked_observation.flatten()

        network_output = neural_network.activate(image_array)
        action_index = np.argmax(network_output)

        observation, reward, done, info = env.step(action_index)

        rew = reward if not isinstance(reward, np.generic) else reward.item()

        current_x = info.get('x_pos', 0)
        if not hasattr(genome, 'prev_x'):
            genome.prev_x = current_x

        speed = current_x - genome.prev_x
        genome.prev_x = current_x

        """
        fitness_bonus = max(current_x ** 1.8 - info.get('time') ** 1.5 +
                            min(max(current_x-50, 0), 1) * 2500 + success * 1e6, 0.00001) * 1e-6
        """

        fitness_bonus = (current_x * (speed ** 2) * 1e-4) if speed > 0 else -0.1
        genome.fitness += rew + fitness_bonus

        if current_x == xpos_max:
            genome.fitness -= 0.05

        if current_x > xpos_max:
            genome.fitness += 1
            xpos_max = current_x

        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            counter = 0
        else:
            counter += 1

        distance_traveled = current_x
        if info.get('flag_get', False):  # Success if the flag is reached
            success = True
            done = True

        life = info.get('life', 0)
        if (life < 2 and genome.fitness < 1500) or counter == 150:
            done = True

    if genome.fitness > best_individual_fitness:
        best_individual_fitness = genome.fitness
        best_individual_stats = {
            "distance": distance_traveled,
            "time": frame,
            "fitness": genome.fitness,
        }
        print(best_individual_stats)

    if success:
        with success_counter.get_lock():
            success_counter.value += 1

    return genome.fitness  # Only return fitness

def preprocess(ob, inx, iny):
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    return ob

def print_info(gene_id: int, gene_fitness: float, info: dict) -> None:
    print('END OF GENOME: ', gene_id)
    print('FITNESS: ', gene_fitness)

    gene_fitness += info.get('score', 0)
    print('FITNESS + SCORE: ', gene_fitness)


def run_mario(config_file, total_iterations: int = 50):

    num_cores = multiprocessing.cpu_count()

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    global generation

    # parallel evaluation is already implemented in NEAT via ParallelEvaluator

    parallel_evaluator = neat.ParallelEvaluator(num_cores, eval_genome)

    for generation in range(1, total_iterations + 1):
        print(f"Now in generation {generation} inside NEAT")
        population.run(parallel_evaluator.evaluate, total_iterations)

        total_genomes = len(population.population)
        success_rate = success_counter.value / total_genomes

        print(f"Generation {generation}: Success Rate = {success_rate:.2%}")

        # save every 100 generations
        if generation % 100 == 0:
            print(f"Saving winner of generation {generation}...")
            best_genome = stats.best_genomes(1)[0]
            save_winner(best_genome, generation)

        with success_counter.get_lock():
            success_counter.value = 0

    winner = stats.best_genomes(1)[0]
    with open('../models/final_winner', 'wb') as output:
        pickle.dump(winner, output, 1)
    return winner.fitness

def save_winner(genome, generation_number):
    """Save the winner genome to a file."""
    filename = f'../models/winner_gen_{generation_number}.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(genome, output, 1)
    print(f"Winner of generation {generation_number + 1} saved to {filename}.")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    # new_config = 'config1'

    MAX_GENERATION_COUNT = 500
    run_mario(config_path, MAX_GENERATION_COUNT)
