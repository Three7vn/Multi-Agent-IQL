import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from datetime import datetime

from .environment import MultiAgentGridworldEnv
from .iql_agent import IQLAgent

def train(args):
    # Create results directory if it doesn't exist
    if not os.path.exists(args.results_dir):
        os.makedirs(args.results_dir)
    
    # Environment Configuration
    env_config = {
        'grid_size': (args.grid_size, args.grid_size), 
        'num_agents': args.num_agents,
        'agent_start_pos': {0: (0, 0), 1: (args.grid_size-1, args.grid_size-1)},
        'agent_goal_pos': {0: (args.grid_size-1, args.grid_size-1), 1: (0, 0)},
        'obstacles_pos': [(args.grid_size//2, args.grid_size//2)], 
        'max_steps': args.max_steps,
        'observation_type': args.observation_type, 
        'reward_type': args.reward_type,
        'slip_prob': args.slip_prob
    }
    
    env = MultiAgentGridworldEnv(**env_config)
    
    # Determine observation dimension based on env config
    if env.observation_type == 'coords':
        obs_dim = 2
    elif env.observation_type == 'local_grid_3x3':
        obs_dim = 9
    elif env.observation_type == 'full_state':
        obs_dim = env.grid_width * env.grid_height
    else:
        raise ValueError("Unsupported observation type")
    
    action_dim = env.action_space_size
    
    # Create agents
    agents = {i: IQLAgent(i, obs_dim, action_dim, args.buffer_capacity, args.learning_rate,
                          args.gamma, args.epsilon_start, args.epsilon_end, args.epsilon_decay,
                          args.target_update_freq)
             for i in env.agent_ids}
    
    episode_rewards_history = []
    avg_rewards_history = []
    total_steps = 0
    
    print(f"Starting IQL Training for {args.num_episodes} episodes...")
    print(f"Device: {agents[0].device}")
    
    for episode in range(args.num_episodes):
        observations = env.reset()
        episode_rewards = {i: 0 for i in env.agent_ids}
        done = False
        
        while not done:
            total_steps += 1
            # 1. Select action for each agent
            joint_action = {agent_id: agent.select_action(observations[agent_id])
                           for agent_id, agent in agents.items()}
            
            # 2. Step the environment
            next_observations, rewards, dones, info = env.step(joint_action)
            
            # 3. Store experience in each agent's buffer
            for agent_id, agent in agents.items():
                agent.replay_buffer.add(observations[agent_id], joint_action[agent_id],
                                       rewards[agent_id], next_observations[agent_id],
                                       dones[agent_id])
            
            # Update observations
            observations = next_observations
            
            # 4. Perform learning step for each agent
            if total_steps % args.learn_every_n_steps == 0:
                for agent_id, agent in agents.items():
                    agent.learn(args.batch_size)
            
            # Update episode rewards
            for agent_id in env.agent_ids:
                episode_rewards[agent_id] += rewards[agent_id]
            
            # Check if episode is finished
            done = dones['__all__']
            
            # Render environment if required
            if args.render and episode % args.render_freq == 0:
                env.render()
        
        # End of episode
        # Decay epsilon for all agents
        current_epsilon = -1.0
        for agent in agents.values():
            agent.decay_epsilon()
            current_epsilon = agent.epsilon
        
        # Log results
        total_episode_reward = sum(episode_rewards.values())
        episode_rewards_history.append(total_episode_reward)
        
        # Calculate average reward over last 50 episodes
        if len(episode_rewards_history) >= 50:
            avg_reward = np.mean(episode_rewards_history[-50:])
        else:
            avg_reward = np.mean(episode_rewards_history)
        
        avg_rewards_history.append(avg_reward)
        
        if (episode + 1) % args.log_freq == 0:
            print(f"Episode {episode + 1}/{args.num_episodes} | "
                  f"Avg Reward (Last 50): {avg_reward:.2f} | "
                  f"Epsilon: {current_epsilon:.3f}")
    
    print("Training finished.")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(args.results_dir, f"iql_results_{timestamp}")
    
    # Plot and save rewards
    plt.figure(figsize=(10, 5))
    plt.plot(episode_rewards_history, alpha=0.3, label='Episode Rewards')
    plt.plot(avg_rewards_history, label='Avg Rewards (50 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('IQL Training Performance')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{results_path}_rewards.png")
    if args.show_plots:
        plt.show()
    
    # Save model for each agent
    for agent_id, agent in agents.items():
        torch.save(agent.q_network.state_dict(), f"{results_path}_agent{agent_id}.pt")
    
    print(f"Results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Independent Q-Learning Agents')
    
    # Environment parameters
    parser.add_argument('--grid_size', type=int, default=5, help='Size of the grid environment')
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents')
    parser.add_argument('--max_steps', type=int, default=50, help='Maximum steps per episode')
    parser.add_argument('--observation_type', type=str, default='coords', 
                        choices=['coords', 'local_grid_3x3', 'full_state'], 
                        help='Type of observation for agents')
    parser.add_argument('--reward_type', type=str, default='individual',
                        choices=['individual', 'shared', 'sparse'],
                        help='Type of reward function')
    parser.add_argument('--slip_prob', type=float, default=0.0, 
                        help='Probability of agents slipping')
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--buffer_capacity', type=int, default=50000, help='Replay buffer capacity')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount factor')
    parser.add_argument('--epsilon_start', type=float, default=1.0, help='Initial exploration rate')
    parser.add_argument('--epsilon_end', type=float, default=0.05, help='Final exploration rate')
    parser.add_argument('--epsilon_decay', type=float, default=0.99, help='Exploration decay rate')
    parser.add_argument('--target_update_freq', type=int, default=100, 
                        help='Frequency of target network updates')
    parser.add_argument('--learn_every_n_steps', type=int, default=4, 
                        help='Learn every N steps')
    
    # Logging and visualization
    parser.add_argument('--log_freq', type=int, default=50, help='Frequency of logging')
    parser.add_argument('--render', action='store_true', help='Render environment')
    parser.add_argument('--render_freq', type=int, default=100, help='Frequency of rendering')
    parser.add_argument('--results_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--show_plots', action='store_true', help='Show plots after training')
    
    args = parser.parse_args()
    
    train(args) 