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

temperature = 1
checkpoint='../checkpoints/NEAT_checkpoint.pkl'
file_tracker='../logging/fitness_logger_NEAT.txt'

config_file = '../configs/config75'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                     neat.DefaultStagnation, config_file)

def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

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


exists = False
if os.path.exists(checkpoint):
    with open(checkpoint, 'rb') as f:
        print('Successfully opened the checkpoint')
        saved_data = pickle.load(f)
        population = saved_data['population']
        generation_number = saved_data['generation_number']
        population = saved_data['population']
        best_individual_fitness = saved_data['best_individual_fitness']
        success_rate = saved_data['success_rate']
        max_time = saved_data['max_time']
        max_distance = saved_data['max_distance']
        avg_time_per_run = saved_data['avg_time_per_run']
        best_genome_so_far = saved_data.get('best_genome_so_far', None)

        exists = True
else:
    generation_number = 0
    best_individual_fitness = float('-inf')
    success_rate = 0
    avg_time_per_run = 0
    max_time = 0
    max_distance = 0
    best_genome_so_far = None
    population = neat.Population(config)

total_distance = 0
manager = multiprocessing.Manager()
distances = manager.list()
def eval_genome(genome, config, skip_frames=4):
    """Evaluates a single genome in NEAT with skipped frames."""

    global success_rate
    global max_distance
    global total_distance  # Global variable for total distance across generations
    global distances

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
    done = False

    while not done:
        frame += 1
        preprocessed_frame = preprocess(observation, width, height) / 255.0
        image_array = preprocessed_frame.flatten()

        total_reward = 0
        network_output = neural_network.activate(image_array)
        action_probs = softmax(network_output)
        action_index = np.random.choice(len(action_probs), p=action_probs)

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

        # Track the maximum distance
        if current_x > max_distance:
            max_distance = current_x

        # Accumulate distance for the average distance calculation
        total_distance += current_x  # Add the current x_pos to total distance
        distances.append(current_x)

        if genome.fitness > best_fitness:
            best_fitness = genome.fitness
            counter = 0
        else:
            counter += 1

        if info.get('flag_get', False):  # Success if the flag is reached
            genome.fitness += 10000
            success_rate += 1

            with open(file_tracker, 'a') as f:
                f.write(f"success +1\n")

            done = True

        life = info.get('life', 0)
        if (life < 2 and genome.fitness < 1500) or counter == 150:
            done = True

    return genome.fitness


def run_mario(total_iterations, bulk_iterations=25, optimizing=False):
    global best_individual_fitness, generation_number, best_genome_so_far, distances
    num_cores = multiprocessing.cpu_count() - 1

    if not exists:
        max_distance = 0  # Initialize max distance

    if not optimizing:
        population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    parallel_evaluator = neat.ParallelEvaluator(num_cores, eval_genome)

    while generation_number < total_iterations:
        start_time = time.time()

        population.run(parallel_evaluator.evaluate, bulk_iterations)  # Run for one generation at a time
        elapsed_time = time.time() - start_time  # Time for one generation

        current_best = stats.best_genomes(1)[0]

        if best_individual_fitness is None or current_best.fitness > best_individual_fitness:
            best_individual_fitness = current_best.fitness
            best_genome_so_far = current_best

        avg_time_per_run = elapsed_time / len(population.population)
        max_distance = np.max(distances)
        avg_distance = np.percentile(distances, 50)
        distance_q3 = np.percentile(distances, 75)


        # Calculate max distance and average distance for the current generation
        fitness_values = [genome.fitness for genome in stats.most_fit_genomes]
        if fitness_values:
            q2 = np.percentile(fitness_values, 50)  # 50th percentile (median)
            q3 = np.percentile(fitness_values, 75)  # 75th percentile
        else:
            q2 = 0
            q3 = 0

        generation_number += bulk_iterations

        try:
            with open(file_tracker, 'a') as f:
                f.write(f"Generation {generation_number}:\n")
                f.write(f"  Best Fitness: {best_individual_fitness}\n")
                f.write(f"  Mean Fitness: {stats.get_fitness_mean()}\n")
                f.write(f"  50th Percentile (Median): {q2}\n")
                f.write(f"  75th Percentile: {q3}\n")
                f.write(f"  Time for Best Individual: {elapsed_time:.2f} sec\n")
                f.write(f"  Average Time per Population Run: {avg_time_per_run:.2f} sec\n")
                f.write(f"  Max Distance: {max_distance}\n")
                f.write(f"  75% of distances: {distance_q3} \n")
                f.write(f"  Average Distance: {avg_distance:.2f}\n")
        except IOError:
            print(f"Error when trying to write to file {file_tracker}")

        print(f"Saving generation number {generation_number} to {checkpoint}")
        with open(checkpoint, 'wb') as f:
            pickle.dump({
                'population': population,
                'generation_number': generation_number,
                'best_individual_fitness': best_individual_fitness,
                'best_genome_so_far': best_genome_so_far,
                'mean_fitness': stats.get_fitness_mean(),
                'max_time': max_time,
                'avg_time_per_run': avg_time_per_run,
                'success_rate': success_rate,
                'max_distance': max_distance
            }, f)

        del distances[:]

    if not optimizing and best_genome_so_far:
        with open('../models/winner_config75_original.pkl', 'wb') as output:
            pickle.dump(best_genome_so_far, output, 1)

    return best_individual_fitness


def save_winner(genome, generation_number):
    filename = f'../models/winner_gen_{generation_number}.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(genome, output, 1)
    print(f"Winner of generation {generation_number + 1} saved to {filename}.")


if __name__ == "__main__":

    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    base_dir = os.path.join(parent_dir, 'configs')
    file_name = '../configs/config75'
    file_path = os.path.join(base_dir, file_name)

    """
    first 50 generations 19:20 -> 
    """

    MAX_GENERATION_COUNT = 500
    print(f"max fitness = {run_mario(MAX_GENERATION_COUNT, 10)}")
