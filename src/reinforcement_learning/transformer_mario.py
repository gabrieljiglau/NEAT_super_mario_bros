import warnings

import gym_super_mario_bros
import gymnasium as gym
import torch.nn as nn
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from src.utils import TrainAndLoggingCallback, RenderModeWrapper, EnvWrapper, CustomJoypadSpace

warnings.warn = lambda *args, **kwargs: None

#  custom feature extraction classes for different types of observations, allowing you to define how
#  to transform the raw observations from the environment into features suitable for the policy network.
class TransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # converts the multi-dimensional observation tensors (e.g., images) into a 1D tensor suitable
        self.flatten = nn.Flatten()

        self.transformer = nn.Transformer(
            d_model=128,  # original 128
            nhead=4,  # original 4
            num_encoder_layers=2  # original2
        )

        self.final_linear = nn.Linear(128, features_dim)

    def forward(self, observations):
        x = self.flatten(observations)
        x = self.transformer(x.unsqueeze(1))
        x = x.squeeze(1)
        return self.final_linear(x)


policy_kwargs = dict(
    features_extractor_class=TransformerExtractor,
    features_extractor_kwargs=dict(features_dim=256)
)


if __name__ == '__main__':
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = CustomJoypadSpace(env, SIMPLE_MOVEMENT)

    env = RenderModeWrapper(env)

    env = EnvWrapper(env)
    env.transform()

    CHECKPOINT_DIR = './train/'
    LOG_DIR = './logs/'

    callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=LOG_DIR, learning_rate=1e-5,
                n_steps=512)  # n_steps: how many steps will be taken before an update to the network will be made

    model.learn(total_timesteps=1000, callback=callback)
    print('hi')
