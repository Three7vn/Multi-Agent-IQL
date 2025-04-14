# Multi-Agent Independent Q-Learning (IQL) in PyTorch

This repository contains an implementation of Independent Q-Learning (IQL) for multi-agent reinforcement learning using PyTorch. The implementation includes a simple gridworld environment where multiple agents must navigate to their respective goals.

## Features

- Independent Q-Learning implementation with deep Q-networks
- Simple multi-agent gridworld environment
- Configurable environment parameters (grid size, number of agents, obstacles)
- Support for different observation types (coordinates, local grid, full state)
- Various reward functions (individual, shared, sparse)
- Visualization tools for training progress and agent evaluation

## Installation

Clone the repository and install the requirements:

```bash
git clone https://github.com/yourusername/multi-agent-IQL-pytorch.git
cd multi-agent-IQL-pytorch
pip install -r requirements.txt
```

## Project Structure

```
multi-agent-IQL-pytorch/
├── README.md          # Project overview, installation
├── requirements.txt   # PyTorch, numpy, etc.
├── src/
│   ├── iql_agent.py   # IQL implementation
│   ├── environment/   # Gridworld/MultiAgentEnv code
│   └── utils/         # Replay buffer, networks
├── notebooks/         # Jupyter notebooks for demos
└── results/           # Training logs, plots, videos
```

## Usage

### Training from Command Line

You can train the agents using the provided training script:

```bash
python src/train.py --num_episodes 1000 --grid_size 5 --num_agents 2 
```

Additional command-line options:

```
--num_episodes: Number of training episodes
--batch_size: Batch size for training
--learning_rate: Learning rate
--epsilon_start: Initial exploration rate
--epsilon_decay: Exploration decay rate
--render: Enable rendering of the environment
--observation_type: Type of observation (coords, local_grid_3x3, full_state)
--reward_type: Type of reward function (individual, shared, sparse)
```

### Using the Jupyter Notebook

There's also a Jupyter notebook available for interactive exploration:

```bash
jupyter notebook notebooks/iql_demo.ipynb
```

## Environment

The environment is a simple gridworld where:

- Multiple agents navigate on a 2D grid
- Each agent has a unique goal position
- Agents can move in four directions (up, down, left, right) or stay in place
- The environment can contain obstacles
- The episode ends when all agents reach their goals or the maximum number of steps is reached

## Independent Q-Learning (IQL)

IQL is a straightforward extension of single-agent Q-learning to multi-agent settings. Key characteristics:

- Each agent has its own Q-network and learns independently
- Agents don't explicitly model other agents' behavior
- Other agents are treated as part of the environment
- Simple to implement but may struggle with coordination tasks

## Results

The training process will generate reward plots and save the trained models in the `results/` directory.
