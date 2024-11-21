import os

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gym import ObservationWrapper
from gym.vector.utils import spaces
from gym.wrappers import GrayScaleObservation
from gymnasium.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


"""
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, terminated, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class ResizeEnv(gym.ObservationWrapper):
    def __init__(self, env, size):
        gym.ObservationWrapper.__init__(self, env)
        (old_height, old_width, old_num_colors) = env.observation_space.shape
        new_shape = (size, size, old_num_colors)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=new_shape, dtype=np.uint8)

    def observation(self, frame):
        height, width, _ = self.observation_space.shape
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
        if frame.ndim == 2:
            frame = frame[:, :, None]
        return frame

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0

    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.current_x_count = 0
        self.max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):

        # truncated: A boolean indicating if the episode was ended due to a time limit
        # or some other constraint, separate from the usual termination conditions.
        state, reward, done, truncated, info = self.env.step(action)
        reward += max(0, info['x_pos'] - self.max_x)
        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 500
            done = True
            print("GOAL")
        if info["life"] < 2:
            reward -= 500
            done = True
        self.current_score = info["score"]
        self.max_x = max(self.max_x, self.current_x)
        self.current_x = info["x_pos"]
        return state, reward / 10., done, info


class CustomJoypadSpace(JoypadSpace):
    def reset(self, **kwargs):
        # Handle seed and options explicitly
        seed = kwargs.pop('seed', None)
        options = kwargs.pop('options', None)

        # Call the reset of the base environment with the remaining kwargs
        result = super().reset(**kwargs)

        # Return the result ensuring it’s in (obs, info) format
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        return obs, info
"""


class PreprocessObservationWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        original_shape = env.observation_space.shape
        self.width = int(original_shape[0] / 8)
        self.height = int(original_shape[1] / 8)

        # Define the new observation space (flattened grayscale image)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(self.width * self.height,), dtype=np.uint8
        )

    def observation(self, obs):
        # Resize observation
        obs = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)

        # Convert to grayscale
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)

        # Flatten the image to a 1D array
        return obs.flatten()

class TrainAndLoggingCallback(BaseCallback):

    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.save_path(model_path)

        return True
