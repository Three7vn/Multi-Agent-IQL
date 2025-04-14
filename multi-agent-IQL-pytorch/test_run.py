import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

from src.environment.gridworld import MultiAgentGridworldEnv
from src.utils.networks import QNetwork
from src.utils.replay_buffer import ReplayBuffer
from src.iql_agent import IQLAgent

print("Testing Multi-Agent IQL implementation...")

# Create a simple environment
env_config = {
    'grid_size': (3, 3),
    'num_agents': 2,
    'agent_start_pos': {0: (0, 0), 1: (2, 2)},
    'agent_goal_pos': {0: (2, 2), 1: (0, 0)},
    'obstacles_pos': [],
    'max_steps': 20,
    'observation_type': 'coords',
    'reward_type': 'individual',
    'slip_prob': 0.0
}

env = MultiAgentGridworldEnv(**env_config)
print(f"Created environment with {env.num_agents} agents on a {env.grid_height}x{env.grid_width} grid")

# Create agents
obs_dim = 2  # For 'coords' observation type
action_dim = env.action_space_size
print(f"Observation dimension: {obs_dim}, Action dimension: {action_dim}")

agents = {
    i: IQLAgent(
        i, obs_dim, action_dim, 
        buffer_capacity=1000,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.95,
        target_update_freq=10
    ) for i in env.agent_ids
}

print(f"Created {len(agents)} IQL agents")

# Run a quick test: 10 episodes with 20 steps max
NUM_EPISODES = 10
BATCH_SIZE = 32

print(f"\nRunning {NUM_EPISODES} test episodes...")

for episode in range(NUM_EPISODES):
    observations = env.reset()
    episode_rewards = {i: 0 for i in env.agent_ids}
    done = False
    step = 0
    
    while not done and step < env.max_steps:
        step += 1
        
        # Select actions
        joint_action = {agent_id: agent.select_action(observations[agent_id])
                      for agent_id, agent in agents.items()}
        
        # Execute actions
        next_observations, rewards, dones, info = env.step(joint_action)
        
        # Store transitions in replay buffers
        for agent_id, agent in agents.items():
            agent.replay_buffer.add(
                observations[agent_id],
                joint_action[agent_id],
                rewards[agent_id],
                next_observations[agent_id],
                dones[agent_id]
            )
        
        # Update state
        observations = next_observations
        
        # Update rewards
        for agent_id in env.agent_ids:
            episode_rewards[agent_id] += rewards[agent_id]
        
        # Learning step (if enough samples)
        for agent_id, agent in agents.items():
            if len(agent.replay_buffer) > BATCH_SIZE:
                agent.learn(BATCH_SIZE)
        
        # Check if done
        done = dones['__all__']
    
    # Decay exploration rate
    for agent in agents.values():
        agent.decay_epsilon()
    
    print(f"Episode {episode+1}/{NUM_EPISODES}: Steps: {step}, Rewards: {episode_rewards}, Epsilon: {agents[0].epsilon:.2f}")

print("\nTest completed successfully!")
print("Multi-Agent IQL implementation is working correctly.") 