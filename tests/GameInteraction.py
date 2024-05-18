import numpy as np
import gym_super_mario_bros

# import the Joypad wrapper
from nes_py.wrappers import JoypadSpace

# import the simplified controls
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


def generate_random_index(buttons):
    number_of_buttons = len(buttons)
    rng = np.random.default_rng(np.random.default_rng())  # Create a new Generator instance
    return rng.integers(0, number_of_buttons)


if __name__ == '__main__':

    # print(env.buttons())
    # print(SIMPLE_MOVEMENT)
    env = gym_super_mario_bros.make('SuperMarioBros-v1')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    # Reset the environment to start
    state = env.reset()

    num_steps = 4000
    for step in range(num_steps):
        index = generate_random_index(SIMPLE_MOVEMENT)  # Generate a random action index for the current step

        state, reward, done, info = env.step(index)  # Execute the action in the environment
        env.render()  # Render the environment to visualize the actions

        if done:
            break  # End the loop if the game is over

    env.close()  # Close the environment when done


