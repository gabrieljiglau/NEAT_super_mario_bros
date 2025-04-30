import os
import pickle
import cv2
import neat
import gym
import subprocess
import numpy as np
import warnings; warnings.warn = lambda *args,**kwargs: None
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from gym.wrappers import RecordVideo
from nes_py.wrappers import JoypadSpace

# .venv is the correct virtual environment

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
env = JoypadSpace(env, RIGHT_ONLY)
video_dir = '../runs'
os.makedirs(video_dir, exist_ok=True)
video_path = os.path.join(video_dir, "winner_run")

env = RecordVideo(env, video_path, episode_trigger=lambda ep: True)

def caption_video(filename, caption):
    output = filename.replace(".mp4", "_captioned.mp4")
    cmd = [
        "ffmpeg", "-i", filename,
        "-vf", f"drawtext=text='{caption}':fontcolor=white:fontsize=24:x=10:y:10",
        "-codec:a", "copy", output
    ]

    subprocess.run(cmd)


def softmax(x):
    x = np.array(x)
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def preprocess(ob, inx, iny):
    ob = cv2.resize(ob, (inx, iny))
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
    ob = np.reshape(ob, (inx, iny))
    return ob

def run_neat(network_path, config_path, training=True, skip_frames=4):
    with open(network_path, 'rb') as f:
        c = pickle.load(f)

    if not training:
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

        if not training:
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

    env.close()

    if not training:
        print(f"distance_travelled = {distance_travelled}")
        print(f"total_reward = {total_reward}")

    if distance_travelled > 1000:
        raw_video_path = os.path.join(video_dir, "good_distance", "episode1.mp4")
        if os.path.exists(raw_video_path):
            caption_video(raw_video_path, f"Distance: {int(distance_travelled)}")

    return distance_travelled

def run_neat_iteratively(network_path, config_path, training=True, sample_size=15):

    total_distance = 0
    for i in range(sample_size):
        total_distance += run_neat(network_path, config_path, training=training)
    return total_distance / sample_size


if __name__ == '__main__':
    # reteaua gasita de meta algoritm: network_path = "../models/winner_config75_original.pkl"
    # run_neat("../models/winner_config75_original.pkl", '../configs/config75', training=False)
    medium_distance = run_neat_iteratively('../models/winner.pkl', '../configs/config75', training=False)
    print(f"mean_distance = {medium_distance}")
