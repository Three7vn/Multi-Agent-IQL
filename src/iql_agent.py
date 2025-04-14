import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque, namedtuple

from .utils.replay_buffer import ReplayBuffer, Transition
from .utils.networks import QNetwork

# --- IQL Agent ---
class IQLAgent:
    """Independent Q-Learning Agent using DQN."""
    def __init__(self, agent_id, observation_dim, action_dim, buffer_capacity=10000,
                 learning_rate=1e-4, gamma=0.99, epsilon_start=1.0,
                 epsilon_end=0.05, epsilon_decay=0.995, target_update_freq=10):
        """
        Initializes the IQL Agent.
        Args:
            agent_id (int): Unique identifier for the agent.
            observation_dim (int): Dimensionality of the observation space.
            action_dim (int): Dimensionality of the action space.
            buffer_capacity (int): Capacity of the replay buffer.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon_start (float): Initial exploration rate.
            epsilon_end (float): Final exploration rate.
            epsilon_decay (float): Decay factor for exploration rate per episode.
            target_update_freq (int): Frequency (in steps) to update the target network.
        """
        self.agent_id = agent_id
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.learn_step_counter = 0 # For target network updates

        # Use GPU if available, otherwise CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize Q-Network and Target Q-Network
        self.q_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_q_network = QNetwork(observation_dim, action_dim).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict()) # Initialize target weights
        self.target_q_network.eval() # Target network is not trained directly

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Replay Buffer for this agent
        self.replay_buffer = ReplayBuffer(buffer_capacity)

    def select_action(self, observation):
        """
        Selects an action using epsilon-greedy strategy.
        Args:
            observation (np.ndarray or torch.Tensor): Current observation for this agent.
        Returns:
            int: The selected action.
        """
        # Convert observation to tensor and move to device if necessary
        if isinstance(observation, np.ndarray):
            observation = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
        elif torch.is_tensor(observation):
             # Ensure it's on the correct device and has batch dimension
             observation = observation.to(self.device)
             if observation.dim() == 1:
                 observation = observation.unsqueeze(0)
        else:
             raise TypeError("Observation must be a numpy array or torch tensor.")

        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Explore: select a random action
            return random.randrange(self.action_dim)
        else:
            # Exploit: select the action with the highest Q-value
            with torch.no_grad(): # No need to track gradients here
                q_values = self.q_network(observation)
                action = q_values.max(1)[1].item() # Get the index of the max Q-value
            return action

    def learn(self, batch_size):
        """
        Performs a learning step by sampling from the replay buffer.
        Args:
            batch_size (int): Number of transitions to sample.
        """
        if len(self.replay_buffer) < batch_size:
            return # Not enough samples yet

        # Sample a batch of transitions
        transitions = self.replay_buffer.sample(batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for details)
        batch = Transition(*zip(*transitions))

        # Concatenate batch elements into tensors
        obs_batch = torch.cat(batch.observation).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        next_obs_batch = torch.cat(batch.next_observation).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device) # Represents (1 - done) effectively

        # --- Calculate Target Q-values ---
        # 1. Get Q-values for next states from the target network
        # We detach because we don't want gradients flowing back through the target network
        next_q_values_target = self.target_q_network(next_obs_batch).detach()
        # 2. Select the maximum Q-value for each next state (greedy part of Q-learning)
        max_next_q_values = next_q_values_target.max(1)[0]
        # 3. Compute the target Q-value: R + gamma * max_a' Q_target(s', a')
        # If the state was terminal (done=1), the target is just the reward.
        # done_batch is 0 if not done, 1 if done. So (1 - done_batch) is 1 if not done, 0 if done.
        target_q_values = reward_batch + (self.gamma * max_next_q_values * (1 - done_batch))
        # Ensure target_q_values has the shape [batch_size, 1]
        target_q_values = target_q_values.unsqueeze(1)

        # --- Calculate Current Q-values ---
        # 1. Get Q-values for the current states and performed actions from the main Q-network
        current_q_values_all = self.q_network(obs_batch)
        # 2. Select the Q-value corresponding to the action actually taken
        current_q_values = current_q_values_all.gather(1, action_batch) # Gathers values along dim 1 using action_batch as indices

        # --- Compute Loss ---
        # Mean Squared Error (MSE) loss between target and current Q-values
        # Equivalent to Bellman error for Q-values [37, 38]
        loss = F.mse_loss(current_q_values, target_q_values)

        # --- Optimize the Model ---
        self.optimizer.zero_grad() # Clear previous gradients
        loss.backward()           # Compute gradients
        # Optional: Clip gradients to prevent exploding gradients
        # torch.nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()      # Update network weights

        # --- Update Target Network ---
        # Periodically copy weights from Q-network to Target Q-network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())

    def decay_epsilon(self):
        """Decays the exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) 