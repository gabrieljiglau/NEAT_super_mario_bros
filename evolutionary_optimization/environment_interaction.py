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

best_individual_fitness = float('-inf')
best_individual_stats = {"distance": 0, "time": 0, "fitness": 0}

def eval_genomes(genomes, config):
    global generation, best_individual_fitness, best_individual_stats
    generation += 1

    log_file = "training_statistics.txt"
    first_time = True

    total_distance = 0
    total_fitness = 0
    total_time = 0

    with open(log_file, "a") as log:
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

                # preprocess observation
                observation = cv2.resize(observation, (width, height))
                observation = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
                observation = np.reshape(observation, (width, height))
                image_array = convert_matrix_into_array(observation, image_array)

                network_output = neural_network.activate(image_array)
                action_index = np.argmax(network_output)

                # previously with 3 output nodes as binary outputs
                """
                network_output = [max(0, min(1, x)) for x in network_output]
                binary_string = "".join(str(round(x)) for x in network_output)
                action_index = int(binary_string, 2)
                if action_index == 7:
                    action_index = 0
                """

                observation, reward, done, info = env.step(action_index)

                # update fitness
                rew = reward if not isinstance(reward, np.generic) else reward.item()
                genome.fitness += rew

                xpos = info.get('x_pos', 0)

                # penalize lack of progress
                if xpos == xpos_max:
                    genome.fitness -= 0.01

                if xpos > xpos_max:
                    genome.fitness += 1
                    xpos_max = xpos

                if genome.fitness > best_fitness:
                    best_fitness = genome.fitness
                    counter = 0
                else:
                    counter += 1

                distance_traveled = xpos
                if info.get('flag_get', False):  # success if the flag reached
                    success += 1
                    done = True

                # end if life is lost or counter exceeds the threshold
                life = info.get('life', 0)
                if life < 2 or counter == 150:
                    done = True

            # update per-generation statistics
            time_taken = frame
            total_distance += distance_traveled
            total_fitness += genome.fitness
            total_time += time_taken

            # check for the best individual
            if genome.fitness > best_individual_fitness:
                best_individual_fitness = genome.fitness
                best_individual_stats = {
                    "distance": distance_traveled,
                    "time": time_taken,
                    "fitness": genome.fitness,
                }

            print_info(gene_id, genome.fitness, info)

        # statistics every 2 generations
        if generation % 2 == 0:
            mean_distance = total_distance / len(genomes)
            mean_fitness = total_fitness / len(genomes)
            mean_time = total_time / len(genomes)

            log.write(f"\n=== Statistics for Generation {generation} ===\n")
            log.write(f"Mean Distance: {mean_distance:.2f}\n")
            log.write(f"Mean Fitness: {mean_fitness:.2f}\n")
            log.write(f"Mean Time: {mean_time:.2f}\n")
            log.write(f"Best Individual - Distance: {best_individual_stats['distance']}, "
                      f"Time: {best_individual_stats['time']}, "
                      f"Fitness: {best_individual_stats['fitness']}\n\n")
            log.flush()

        # success rate for this generation
        log.write(f"Success rate: {success / len(genomes):.2f}\n")


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


def run_mario(config_file, total_iterations: int = 50):

    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet,
                         neat.DefaultStagnation, config_file)
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    global generation

    for gen in range(1, total_iterations + 1):
        population.run(eval_genomes, 1)

        # save every 100 generations
        if generation % 100 == 0:
            print(f"Saving winner of generation {generation}...")
            best_genome = stats.best_genomes(1)[0]
            save_winner(best_genome, generation)

    winner = stats.best_genomes(1)[0]
    with open('../../models/final_winner', 'wb') as output:
        pickle.dump(winner, output, 1)
    return stats.best_genomes(1)[0]

def save_winner(genome, generation_number):
    """Save the winner genome to a file."""
    filename = f'../../models/winner_gen_{generation_number}.pkl'
    with open(filename, 'wb') as output:
        pickle.dump(genome, output, 1)
    print(f"Winner of generation {generation_number + 1} saved to {filename}.")


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config")
    new_config = 'config1'

    MAX_GENERATION_COUNT = 2000
    run_mario(config_path, MAX_GENERATION_COUNT)
