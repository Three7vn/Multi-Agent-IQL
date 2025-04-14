import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Q-Network ---
class QNetwork(nn.Module):
    """Simple MLP Q-Network for IQL."""
    def __init__(self, observation_dim, action_dim):
        """
        Initializes the Q-Network.
        Args:
            observation_dim (int): Dimensionality of the agent's observation.
            action_dim (int): Number of possible actions for the agent.
        """
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(observation_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, action_dim)

    def forward(self, obs):
        """
        Forward pass through the network.
        Args:
            obs (torch.Tensor): Batch of observations.
        Returns:
            torch.Tensor: Q-values for each action.
        """
        # Ensure input is float
        if obs.dtype != torch.float32:
           obs = obs.float()
        x = F.relu(self.layer1(obs))
        x = F.relu(self.layer2(x))
        return self.layer3(x) 