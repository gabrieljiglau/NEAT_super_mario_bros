import numpy as np
import matplotlib.pyplot as plt
from gymnasium import Wrapper

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i - 100):(i + 1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


class CustomMarioWrapper(Wrapper):
    def __init__(self, env):
        # Initialize the Wrapper superclass
        super().__init__(env)  # This line is equivalent to gym.Wrapper.__init__(self, env)

        # Custom initialization here, if needed
        self.env = env


"""
    def reset(self, **kwargs):
        # Reset the environment and add custom reset behavior if needed
        obs, info = self.env.reset(**kwargs)
        # Add custom code here, if desired
        return obs, info

    def step(self, action):
        # Modify the step method to add custom functionality
        obs, reward, done, truncated, info = self.env.step(action)
        # Add custom code here, if desired
        return obs, reward, done, truncated, info
"""
