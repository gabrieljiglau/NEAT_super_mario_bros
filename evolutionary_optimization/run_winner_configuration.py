import os
import pickle
import cv2
import neat
import numpy as np
import warnings; warnings.warn = lambda *args,**kwargs: None
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)

def softmax(x, temp=1.0):
    """computes softmax values for each output"""
    x = np.array(x)
    x = x / temp
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def preprocess(ob, inx, iny):
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    return ob

def run_neat(skip_frames=4):
    with open("../models/final_winner.pkl", 'rb') as f:
        c = pickle.load(f)

    print('Loaded genome:')
    print(c)

    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, '../tests/config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    neural_network = neat.nn.FeedForwardNetwork.create(c, config)

    inx, iny, inc = env.observation_space.shape
    inx = int(inx / 8)
    iny = int(iny / 8)

    observation = env.reset()

    done = False

    distance_travelled = 0
    total_reward = 0
    frame = 0
    action_index = 0

    while not done:
        env.render()

        frame += 1
        processed_ob = preprocess(observation, inx, iny)
        print(f"processed observation = {processed_ob}")
        image_array = processed_ob.flatten()

        """
        nn_output = [max(0, min(1, x)) for x in nn_output]
        binary_string = "".join(str(round(x)) for x in nn_output)
        int_output = int(binary_string, 2)

        num_actions = len(SIMPLE_MOVEMENT)
        int_output = int_output % num_actions
        """

        if frame % skip_frames == 0:
            network_output = neural_network.activate(image_array)
            action_probs = softmax(network_output, 1)
            action_index = np.random.choice(len(action_probs), p=action_probs)
        # print(f"info = {info}")

        total_reward = 0
        for _ in range(skip_frames):
            observation, reward, done, info = env.step(action_index)
            total_reward += reward

            if done:  # episode might end early
                break

        rew = reward if not isinstance(reward, np.generic) else reward.item()
        # print(f"reward = {rew}")

        x_pos = info.get('x_pos', 0)
        # print(f"x_pos = {x_pos}")

        distance_travelled = x_pos
        total_reward += rew

    print(f"distance_travelled = {distance_travelled}")
    print(f"total_reward = {total_reward}")


if __name__ == '__main__':
    run_neat()
