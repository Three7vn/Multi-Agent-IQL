import torch
import random
import numpy as np
from collections import deque, namedtuple

# --- Replay Buffer ---
# Standard replay buffer implementation
Transition = namedtuple('Transition',
                        ('observation', 'action', 'reward', 'next_observation', 'done'))

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""
    def __init__(self, capacity):
        """
        Initializes the Replay Buffer.
        Args:
            capacity (int): Maximum number of transitions to store.
        """
        self.memory = deque(maxlen=capacity)

    def add(self, obs, action, reward, next_obs, done):
        """Adds a transition to the buffer."""
        # Ensure action is stored as a tensor for batching
        action_tensor = torch.tensor([[action]], dtype=torch.long)
        # Ensure others are tensors or appropriate types
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.float32) # Use float for multiplication later

        # Convert numpy observations to tensors if they aren't already
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).float().unsqueeze(0)
        if isinstance(next_obs, np.ndarray):
            next_obs = torch.from_numpy(next_obs).float().unsqueeze(0)

        self.memory.append(Transition(obs, action_tensor, reward_tensor, next_obs, done_tensor))

    def sample(self, batch_size):
        """Samples a batch of transitions randomly."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.memory) 