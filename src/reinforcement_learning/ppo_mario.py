import warnings

import gym_super_mario_bros

warnings.warn = lambda *args, **kwargs: None
import gymnasium as gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from src.reinforcement_learning.plots import CustomMarioWrapper

# Create and wrap the Super Mario environment
env = gym_super_mario_bros.make('SuperMarioBros-v0')
env = JoypadSpace(env, SIMPLE_MOVEMENT)  # Apply JoypadSpace for compatible action space
env = CustomMarioWrapper(env)  # Apply custom wrapper

# exemplu la baiatul cu mario din playlist-ul 'EI AI'

CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'


class CustomTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        # Flatten the input observation shape to fit the transformer model requirements
        self.flatten = nn.Flatten()

        # Define the transformer
        self.transformer = nn.Transformer(
            d_model=128,  # Adjust dimensions based on your input
            nhead=4,
            num_encoder_layers=2
        )

        # Optional final linear layer to get the exact feature dimension
        self.final_linear = nn.Linear(128, features_dim)

    def forward(self, observations):
        # Flatten observations and add dimension adjustments if needed
        x = self.flatten(observations)
        x = self.transformer(x.unsqueeze(1))  # Adjust for transformer input shape
        x = x.squeeze(1)  # Remove unnecessary dimensions
        return self.final_linear(x)


policy_kwargs = dict(
    features_extractor_class=CustomTransformerExtractor,
    features_extractor_kwargs=dict(features_dim=256)
)

# Initialize the PPO model with the custom transformer
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)

# Train the model
# model.learn(total_timesteps=10000)

if __name__ == '__main__':
    print('hi')
