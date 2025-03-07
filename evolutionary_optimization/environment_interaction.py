import os
import time
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
"""

STACK_SIZE = 4
frame_stack = deque(maxlen=STACK_SIZE)
env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

success = 0
generation = 0
temperature = 1

best_individual_fitness = float('-inf')
best_individual_stats = {"distance": 0, "time": 0, "fitness": 0}

def softmax(x, temp=1.0):
    x = np.array(x)
    x = x / temp
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def eval_genome(genome, config, skip_frames=4, STACK_SIZE=4):
    """Evaluates a single genome in NEAT with stacked frames."""
    global generation, best_individual_fitness, best_individual_stats, success

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
    action_index = 0

    # Fill the frame stack with the first preprocessed frame
    preprocessed_frame = preprocess(observation, width, height) / 255.0  # Normalize
    for _ in range(STACK_SIZE):
        frame_stack.append(preprocessed_frame)

    while not done:
        frame += 1

        # Preprocess the observation
        preprocessed_frame = preprocess(observation, width, height) / 255.0  # Normalize
        frame_stack.append(preprocessed_frame)  # Automatically removes the oldest frame when full

        # Ensure we only start once we have a full stack
        if len(frame_stack) < STACK_SIZE:
            continue

        # Stack the frames along the depth dimension and flatten
        stacked_observation = np.stack(frame_stack, axis=0)
        image_array = stacked_observation.flatten()

        # Perform action selection every `skip_frames` steps
        if frame % skip_frames == 0:
            network_output = neural_network.activate(image_array)
            action_probs = softmax(network_output, temperature)
            action_index = np.random.choice(len(action_probs), p=action_probs)

        total_reward = 0
        for _ in range(skip_frames):
            observation, reward, done, info = env.step(action_index)
            total_reward += reward
            if done:  # Episode might end early
                break

        rew = total_reward if not isinstance(total_reward, np.generic) else total_reward.item()

        current_x = info.get('x_pos', 0)
        if not hasattr(genome, 'prev_x'):
            genome.prev_x = current_x

        speed = current_x - genome.prev_x
        genome.prev_x = current_x

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
            genome.fitness += 10000
            done = True
            success += 1

        life = info.get('life', 0)
        if (life < 2 and genome.fitness < 1500) or counter == 150:
            done = True

    if generation % 5 == 0:
        with open('training_statistics.txt', 'a') as file:
            file.write(f"Now in generation {generation}; best_fintess {best_fitness}; best_distance {xpos_max}; "
                       f"success = {success}")

    if genome.fitness > best_individual_fitness:
        best_individual_fitness = genome.fitness
        best_individual_stats = {
            "distance": distance_traveled,
            "time": frame,
            "fitness": genome.fitness,
        }

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


def run_mario(config_file, total_iterations, optimizing=False):
    num_cores = multiprocessing.cpu_count() - 1

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_file)
    population = neat.Population(config)

    if not optimizing:
        population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    global generation

    # parallel evaluation is already implemented in NEAT via ParallelEvaluator
    parallel_evaluator = neat.ParallelEvaluator(num_cores, eval_genome)

    best_fitness_so_far = None
    best_genome_so_far = None

    for gen in range(total_iterations):
        start_time = time.time()

        population.run(parallel_evaluator.evaluate, 1)  # run for one generation at a time

        elapsed_time = time.time() - start_time

        current_best = stats.best_genomes(1)[0]

        if best_fitness_so_far is None or current_best.fitness > best_fitness_so_far:
            best_fitness_so_far = current_best.fitness
            best_genome_so_far = current_best

        if elapsed_time > 300:
            print("Timeout reached (300s)! Returning best fitness so far.")
            break

    if not optimizing and best_genome_so_far:
        with open('../models/winner_config75.pkl', 'wb') as output:
            pickle.dump(best_genome_so_far, output, 1)
            print(f"success rate = {float(success/total_iterations)}")

    return best_fitness_so_far if best_genome_so_far else None

def save_winner(genome, generation_number):
    """Save the winner genome to a file."""
    filename = f'../models/winner_gen_{generation_number}.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(genome, output, 1)
    print(f"Winner of generation {generation_number + 1} saved to {filename}.")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    new_config = 'config'

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    base_dir = os.path.join(parent_dir, 'configs')
    file_name = 'config75'
    file_path = os.path.join(base_dir, file_name)

    MAX_GENERATION_COUNT = 100
    print(f"max fitness = {run_mario(file_path, MAX_GENERATION_COUNT)}")
