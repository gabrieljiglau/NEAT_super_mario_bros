import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical


class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)

        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
            np.array(self.actions), \
            np.array(self.probs), \
            np.array(self.vals), \
            np.array(self.rewards), \
            np.array(self.dones), \
            batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []


class TransformerActorNetwork(nn.Module):
    def __init__(self, num_actions, input_dims, alpha, d_model=128, nhead=4, num_layers=2, chkpt_dir='tmp'):
        super(TransformerActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')

        # Assuming input_dims = (30, 32), we flatten the input to (960)
        self.fc_in = nn.Linear(input_dims[0], d_model)  # input_dims[0] * input_dims[1] = 960

        # Transformer encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead),
            num_layers=num_layers
        )

        # Final fully connected layer for output (num_actions)
        self.fc_out = nn.Linear(d_model, num_actions)

        # Optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        # Device configuration
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Flatten the state input to (batch_size, 960)
        state = state.view(state.size(0), -1)  # Flatten to (batch_size, 960)
        x = self.fc_in(state)  # Input is now (batch_size, 960)

        # Prepare for transformer input (add sequence dimension)
        x = x.unsqueeze(1)  # Transformer expects (seq_len, batch, embedding_dim)

        # Pass through transformer
        transformer_out = self.transformer(x)

        # Take the output, project to action space
        x = transformer_out.squeeze(1)  # Remove sequence dimension
        x = self.fc_out(x)

        # Apply softmax to get action probabilities
        action_probs = T.softmax(x, dim=-1)

        return Categorical(action_probs)

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class CriticNetwork(nn.Module):
    # alpha: learning_rate
    def __init__(self, input_dims, alpha, fc1_dims=256, fc2_dims=256, chkpt_dir='tmp'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')
        self.critic = nn.Sequential(
            nn.Linear(*input_dims, fc1_dims),
            nn.ReLU(),
            nn.Linear(fc1_dims, fc2_dims),
            nn.ReLU(),
            nn.Linear(fc2_dims, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda: 0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    # gamma: discount factor
    # alpha: learning rate
    # N: the horizon (the number of steps before an update is performed)
    def __init__(self, num_actions, input_dims, gamma=0.99, alpha=0.0003, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, num_epochs=10):
        self.gamma = gamma
        self.policy_clip = policy_clip
        self.num_epochs = num_epochs
        self.gae_lambda = gae_lambda

        self.actor = TransformerActorNetwork(num_actions, input_dims, alpha)
        self.critic = CriticNetwork(input_dims, alpha)
        self.memory = PPOMemory(batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving model ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading model ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, observation):
        # Ensure observation is a numpy array and correctly reshaped
        observation = np.array(observation)

        # Convert observation to a PyTorch tensor and ensure the shape is compatible with the input layer
        state = T.tensor(observation, dtype=T.float).to(self.actor.device)

        # Flatten state to (1, 960) - input layer expects flattened observation
        state = state.view(1, -1)  # Flatten the observation to (1, 960)

        # Pass through actor to get action distribution
        distribution = self.actor(state)  # Input is now (1, 960)
        action = distribution.sample()

        # Pass through critic (same flattened state)
        value = self.critic(state)

        # Extract log probabilities and values
        probs = T.squeeze(distribution.log_prob(action)).item()
        action = T.squeeze(action).item()
        value = T.squeeze(value).item()

        return action, probs, value

    def learn(self):
        for _ in range(self.num_epochs):
            state_arr, action_arr, old_probs_arr, vals_arr, \
                reward_arr, done_arr, batches = self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr) - 1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * (1 - int(done_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda

                advantage[t] = a_t

            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_probs_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).to(self.actor.device)

                distribution = self.actor(states)
                critic_value = self.critic(states)
                critic_value = T.squeeze(critic_value)

                new_probs = distribution.log_prob(actions)

                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clip_probs = T.clamp(prob_ratio, 1 - self.policy_clip, 1 + self.policy_clip) * advantage[batch]

                actor_loss = -T.min(weighted_probs, weighted_clip_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value)**2  # squared
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss

                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                total_loss.backward()

                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
