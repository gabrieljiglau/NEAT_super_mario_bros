import numpy as np
import cv2
import neat
import pickle
import os
import warnings; warnings.warn = lambda *args,**kwargs: None
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

generation = 0

def eval_genomes(genomes, config):

    global generation
    generation += 1

    log_file = "training_statistics.txt"  # File to log statistics
    first_time = True

    with open(log_file, "a") as log:  # Open in append mode

        if first_time:
            log.write(f"Current generation: {generation} \n")
            first_time = False

        for gene_id, genome in genomes:
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
            success = False

            while not done:
                image_array = []
                frame += 1

                # Preprocess observation
                observation = cv2.resize(observation, (width, height))
                observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
                observation = np.reshape(observation, (width, height))
                image_array = convert_matrix_into_array(observation, image_array)

                # Neural network decision
                network_output = neural_network.activate(image_array)
                network_output = [max(0, min(1, x)) for x in network_output]
                binary_string = "".join(str(round(x)) for x in network_output)
                int_output = int(binary_string, 2)
                if int_output == 7:
                    int_output = 0

                # Environment step
                observation, reward, done, info = env.step(int_output)

                # Update fitness
                rew = reward if not isinstance(reward, np.generic) else reward.item()
                genome.fitness += rew

                xpos = info.get('x_pos', 0)
                if xpos > xpos_max:
                    genome.fitness += 1
                    xpos_max = xpos

                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    counter = 0
                else:
                    counter += 1

                # Track distance and success status
                distance_traveled = xpos
                if info.get('flag_get', False):  # Success if the flag reached
                    success = True
                    done = True

                # End if life is lost or counter exceeds the threshold
                life = info.get('life', 0)
                if life < 2 or counter == 150:
                    done = True

            # Log statistics for this genome
            time_taken = frame

            log.write(
                f"Genome ID: {gene_id}, Distance: {distance_traveled}, Time: {time_taken}, "
                f"Success: {success}, Fitness: {genome.fitness}\n"
            )
            log.flush()  # Ensure logs are written to file

            # Print genome summary to console
            print_info(gene_id, genome.fitness, info)


def convert_matrix_into_array(observation, image_array):
    for x in observation:
        for y in x:
            image_array.append(y)

    return image_array

def print_info(gene_id: int, gene_fitness: float, info: dict) -> None:
    print('END OF GENOME: ', gene_id)
    print('FITNESS: ', gene_fitness)

    gene_fitness += info.get('score', 0)
    print('FITNESS + SCORE: ', gene_fitness)


def run(config_file, total_iterations: int = 100) -> None:
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    winner = population.run(eval_genomes, total_iterations)

    with open('../../tests/winner', 'wb') as output:
        pickle.dump(winner, output, 1)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "../../tests/evolutionary_optimization/config")

    MAX_GENERATION_COUNT = 1
    run(config_path, MAX_GENERATION_COUNT)
