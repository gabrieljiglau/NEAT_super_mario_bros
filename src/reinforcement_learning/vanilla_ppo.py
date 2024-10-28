import cv2
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import numpy as np
from src.reinforcement_learning.algorithm import Agent
from src.reinforcement_learning.plots import plot_learning_curve


# Preprocess function to resize and convert the observation to grayscale
def preprocess(ob, inx, iny):
    ob = cv2.resize(ob, (inx, iny))  # Resize image
    ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    ob = np.reshape(ob, (inx, iny))  # Reshape to required dimensions
    ob = ob / 255.0  # Normalize pixel values between 0 and 1
    return ob


if __name__ == '__main__':

    # Initialize Super Mario environment with SIMPLE_MOVEMENT
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)

    N = 20  # Horizon
    batch_size = 5
    num_epochs = 100
    alpha = 0.003  # Learning rate

    # Get observation space shape and preprocess size
    inx, iny, inc = env.observation_space.shape
    inx = int(inx / 8)  # Downscale by a factor (e.g., 8)
    iny = int(iny / 8)  # Downscale by a factor (e.g., 8)
    input_dims = (inx, iny)  # PPO Agent input dimensions after preprocessing

    # Initialize PPO Agent
    agent = Agent(num_actions=env.action_space.n, batch_size=batch_size,
                  alpha=alpha, num_epochs=num_epochs,
                  input_dims=input_dims)

    num_games = 100
    best_score = env.reward_range[0]
    score_history = []
    figure_file = 'plots/cartpole.png'

    learn_iters = 0

    for i in range(num_games):
        observation = env.reset()  # Reset the environment
        observation = preprocess(observation, inx, iny)  # Preprocess observation

        done = False
        accumulated_score = 0
        num_steps = 0

        while not done:
            # env.render()  # Render the environment (optional)

            # Choose action using the agent (feed preprocessed observation)
            action, prob, val = agent.choose_action(observation)

            # Step the environment with the chosen action
            new_state, reward, done, truncated, info = env.step(action)

            # Preprocess new state (next observation)
            new_state = preprocess(new_state, inx, iny)

            accumulated_score += reward
            num_steps += 1

            # Store experience in agent's memory
            agent.remember(observation, action, prob, val, reward, done)

            if num_steps % N == 0:
                # Learn after every N steps
                agent.learn()
                learn_iters += 1

            # Update the current state for the next step
            observation = new_state

        # Log the episode score
        score_history.append(accumulated_score)
        avg_score = np.mean(score_history[-100:])

        # Save the model if a better score is achieved
        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        # Print stats for each episode
        print(f"episode {i}, score {accumulated_score}, avg_score {avg_score},"
              f" time_steps {num_steps}, learning_steps {learn_iters}")

    # Close the environment after training
    env.close()

    x = [i + 1 for i in range(len(score_history))]
    plot_learning_curve(x, score_history, figure_file)
