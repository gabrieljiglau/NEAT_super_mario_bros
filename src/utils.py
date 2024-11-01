import os
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
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


class CustomJoypadSpace(JoypadSpace):
    def reset(self, **kwargs):
        kwargs.pop("seed", None)  # Remove 'seed' if it exists in kwargs
        kwargs.pop("options", None)  # Remove 'options' if it exists in kwargs

        result = super().reset(**kwargs)

        # Ensure result is returned as (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        return obs, info

class RenderModeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def render_mode(self):
        return getattr(self.env, 'render_mode', None)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)

        # Ensure result is returned as (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        return obs, info

    def step(self, action):
        return self.env.step(action)


class CustomDummyVecEnv(DummyVecEnv):
    def reset(self, **kwargs):
        kwargs.pop("seed", None)  # Remove 'seed' if it exists in kwargs
        kwargs.pop("options", None)  # Remove 'options' if it exists in kwargs

        # Call the parent's reset, ensuring we get a tuple (obs, info)
        result = super().reset(**kwargs)

        # Ensure result is returned as (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        return obs, info

class EnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env = env

    def _apply_grayscale(self):
        self.env = GrayScaleObservation(self.env, keep_dim=True)

    def _wrap_env(self):
        self.env = CustomDummyVecEnv([lambda: self.env])

    def _apply_frame_stacking(self):
        self.env = FrameStack(self.env, 4)

    def transform(self):
        self._apply_grayscale()
        self._wrap_env()
        self._apply_frame_stacking()

        return self

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
