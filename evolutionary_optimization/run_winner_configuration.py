import os
import pickle
import cv2
import neat
import numpy as np
import warnings; warnings.warn = lambda *args,**kwargs: None
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)


def preprocess(ob, inx, iny):
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    return ob

def run_neat():
    with open("../models/winner_gen_200_second.pkl", 'rb') as f:
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
    while not done:
        env.render()

        processed_ob = preprocess(observation, inx, iny)
        image_array = processed_ob.flatten()

        """
        nn_output = [max(0, min(1, x)) for x in nn_output]
        binary_string = "".join(str(round(x)) for x in nn_output)
        int_output = int(binary_string, 2)

        num_actions = len(SIMPLE_MOVEMENT)
        int_output = int_output % num_actions
        """

        network_output = neural_network.activate(image_array)
        action_index = np.argmax(network_output)
        observation, reward, done, info = env.step(action_index)


if __name__ == '__main__':
    run_neat()
