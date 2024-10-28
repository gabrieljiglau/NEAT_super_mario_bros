import os

from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
import gym_super_mario_bros


class EnvModifier:

    def __init__(self, level: str):
        self.env = gym_super_mario_bros.make(level)
        self.env = JoypadSpace(self.env, SIMPLE_MOVEMENT)

    def apply_grayscale(self):
        self.env = GrayScaleObservation(self.env, keep_dim=True)

    def wrap_env(self):
        self.env = DummyVecEnv([lambda: self.env])

    def apply_frame_stacking(self):
        self.env = VecFrameStack(self.env, 4, channels_order='last')


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















