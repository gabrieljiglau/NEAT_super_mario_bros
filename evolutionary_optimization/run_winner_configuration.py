import os
import pickle
import cv2
import neat
import numpy as np
import warnings; warnings.warn = lambda *args,**kwargs: None
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

# .venv is the correct virtual environment

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def preprocess(ob, inx, iny):
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    return ob

def run_neat(network_path, config_path, skip_frames=4):
    with open(network_path, 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, config_path)
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    neural_network = neat.nn.FeedForwardNetwork.create(c, config)

    width, height, color = env.observation_space.shape
    width = int(width / 8)
    height = int(height / 8)

    observation = env.reset()

    done = False

    distance_travelled = 0
    total_reward = 0
    frame = 0

    while not done:
        env.render()

        frame += 1
        preprocessed_frame = preprocess(observation, width, height) / 255

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

        rew = reward if not isinstance(reward, np.generic) else reward.item()
        # print(f"reward = {rew}")

        x_pos = info.get('x_pos', 0)
        # print(f"x_pos = {x_pos}")

        distance_travelled = x_pos
        total_reward += rew

    print(f"distance_travelled = {distance_travelled}")
    print(f"total_reward = {total_reward}")
    return distance_travelled

# network_path = "../models/winner_config75_original.pkl"
def run_neat_iteratively(network_path, config_path, sample_size=10):

    total_distance = 0
    for i in range(sample_size):
        total_distance += run_neat(network_path, config_path)
    return total_distance / sample_size


if __name__ == '__main__':

    run_neat("../models/winner_config75_original.pkl", '../configs/config75')
