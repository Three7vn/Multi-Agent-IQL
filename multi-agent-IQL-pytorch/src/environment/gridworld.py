import numpy as np

class MultiAgentGridworldEnv:
    """
    A simple multi-agent gridworld environment where agents must navigate to their goals.
    """
    
    def __init__(self, grid_size=(5, 5), num_agents=2, agent_start_pos=None, 
                 agent_goal_pos=None, obstacles_pos=None, max_steps=50,
                 observation_type='coords', reward_type='individual', slip_prob=0.0):
        """
        Initialize the environment.
        
        Args:
            grid_size (tuple): Size of the grid (height, width)
            num_agents (int): Number of agents in the environment
            agent_start_pos (dict): Starting positions of agents {agent_id: (x, y)}
            agent_goal_pos (dict): Goal positions of agents {agent_id: (x, y)}
            obstacles_pos (list): List of obstacle positions [(x, y)]
            max_steps (int): Maximum number of steps per episode
            observation_type (str): Type of observation ('coords', 'local_grid_3x3', 'full_state')
            reward_type (str): Type of reward function ('individual', 'shared', 'sparse')
            slip_prob (float): Probability of agent slipping (moving in random direction)
        """
        self.grid_height, self.grid_width = grid_size
        self.num_agents = num_agents
        self.agent_ids = list(range(num_agents))
        
        # Default starting and goal positions if not provided
        if agent_start_pos is None:
            self.agent_start_pos = {0: (0, 0), 1: (self.grid_height-1, self.grid_width-1)}
        else:
            self.agent_start_pos = agent_start_pos
            
        if agent_goal_pos is None:
            self.agent_goal_pos = {0: (self.grid_height-1, self.grid_width-1), 1: (0, 0)}
        else:
            self.agent_goal_pos = agent_goal_pos
            
        self.obstacles_pos = obstacles_pos if obstacles_pos is not None else []
        
        # Environment parameters
        self.max_steps = max_steps
        self.observation_type = observation_type
        self.reward_type = reward_type
        self.slip_prob = slip_prob
        
        # Action space: 0: up, 1: right, 2: down, 3: left, 4: stay
        self.action_space_size = 5
        self.actions = [(0, -1), (1, 0), (0, 1), (-1, 0), (0, 0)]  # (dx, dy)
        
        # Initialize environment state
        self.agent_positions = {}
        self.steps = 0
        self.done = False
        
        # Initialize the grid
        self.reset()
        
    def reset(self):
        """
        Reset the environment to initial state.
        
        Returns:
            dict: Initial observations for each agent
        """
        # Reset agent positions
        self.agent_positions = self.agent_start_pos.copy()
        
        # Reset step counter and done flag
        self.steps = 0
        self.done = False
        
        # Return initial observations
        return self._get_observations()
    
    def step(self, actions):
        """
        Take a step in the environment based on the agents' actions.
        
        Args:
            actions (dict): Actions for each agent {agent_id: action}
            
        Returns:
            tuple: (observations, rewards, dones, info)
        """
        if self.done:
            return self._get_observations(), {i: 0 for i in self.agent_ids}, {i: True for i in self.agent_ids}, {}
        
        self.steps += 1
        
        # Move agents one by one
        for agent_id in self.agent_ids:
            if agent_id in actions:
                # Apply slip probability (agent moves in random direction)
                if np.random.random() < self.slip_prob:
                    action = np.random.randint(0, self.action_space_size)
                else:
                    action = actions[agent_id]
                
                # Get movement direction
                dx, dy = self.actions[action]
                
                # Current position
                x, y = self.agent_positions[agent_id]
                
                # New position
                new_x, new_y = x + dx, y + dy
                
                # Check if new position is valid
                if self._is_valid_position(new_x, new_y):
                    # Check for collisions with other agents
                    collision = False
                    for other_id, pos in self.agent_positions.items():
                        if other_id != agent_id and (new_x, new_y) == pos:
                            collision = True
                            break
                    
                    if not collision:
                        self.agent_positions[agent_id] = (new_x, new_y)
        
        # Get observations and rewards
        observations = self._get_observations()
        rewards = self._get_rewards()
        
        # Check if episode is done
        agent_dones = {}
        for agent_id in self.agent_ids:
            agent_dones[agent_id] = self._is_done(agent_id)
        
        # Episode is done if any agent reaches its goal or max steps is reached
        episode_done = all(agent_dones.values()) or self.steps >= self.max_steps
        if episode_done:
            self.done = True
        
        dones = {agent_id: agent_dones[agent_id] for agent_id in self.agent_ids}
        dones['__all__'] = episode_done
        
        # Additional info
        info = {}
        
        return observations, rewards, dones, info
    
    def _get_observations(self):
        """
        Get observations for all agents based on the observation type.
        
        Returns:
            dict: Observations for each agent
        """
        observations = {}
        
        for agent_id in self.agent_ids:
            if self.observation_type == 'coords':
                # Just the agent's (x, y) coordinates
                observations[agent_id] = np.array(self.agent_positions[agent_id])
                
            elif self.observation_type == 'local_grid_3x3':
                # 3x3 grid around the agent
                x, y = self.agent_positions[agent_id]
                local_grid = np.zeros((3, 3), dtype=np.float32)
                
                for i in range(3):
                    for j in range(3):
                        grid_x, grid_y = x + (i-1), y + (j-1)
                        
                        # Check if position is out of bounds
                        if not (0 <= grid_x < self.grid_width and 0 <= grid_y < self.grid_height):
                            # Mark as wall
                            local_grid[i, j] = -1
                        # Check if position has an obstacle
                        elif (grid_x, grid_y) in self.obstacles_pos:
                            local_grid[i, j] = -1
                        # Check if position has another agent
                        else:
                            for other_id, pos in self.agent_positions.items():
                                if other_id != agent_id and pos == (grid_x, grid_y):
                                    local_grid[i, j] = 1
                            
                            # Check if position is the goal
                            if (grid_x, grid_y) == self.agent_goal_pos[agent_id]:
                                local_grid[i, j] = 2
                
                observations[agent_id] = local_grid.flatten()
                
            elif self.observation_type == 'full_state':
                # Full grid representation
                full_grid = np.zeros((self.grid_height, self.grid_width), dtype=np.float32)
                
                # Mark obstacles
                for obs_x, obs_y in self.obstacles_pos:
                    full_grid[obs_y, obs_x] = -1
                
                # Mark goals
                for a_id, (goal_x, goal_y) in self.agent_goal_pos.items():
                    if a_id == agent_id:
                        full_grid[goal_y, goal_x] = 2  # Own goal
                    else:
                        full_grid[goal_y, goal_x] = 3  # Other agent's goal
                
                # Mark agents
                for a_id, (agent_x, agent_y) in self.agent_positions.items():
                    if a_id == agent_id:
                        full_grid[agent_y, agent_x] = 1  # Self
                    else:
                        full_grid[agent_y, agent_x] = 4  # Other agent
                
                observations[agent_id] = full_grid.flatten()
            
        return observations
    
    def _get_rewards(self):
        """
        Calculate rewards for all agents based on the reward type.
        
        Returns:
            dict: Rewards for each agent
        """
        rewards = {}
        
        for agent_id in self.agent_ids:
            current_pos = self.agent_positions[agent_id]
            goal_pos = self.agent_goal_pos[agent_id]
            
            if self.reward_type == 'individual':
                # Negative reward based on distance to goal
                manhattan_dist = abs(current_pos[0] - goal_pos[0]) + abs(current_pos[1] - goal_pos[1])
                rewards[agent_id] = -0.1 * manhattan_dist
                
                # Bonus for reaching goal
                if current_pos == goal_pos:
                    rewards[agent_id] += 10.0
                
                # Penalty for collisions with obstacles
                if current_pos in self.obstacles_pos:
                    rewards[agent_id] -= 5.0
                
            elif self.reward_type == 'shared':
                # Sum of all agent rewards
                team_reward = 0
                for a_id in self.agent_ids:
                    a_pos = self.agent_positions[a_id]
                    a_goal = self.agent_goal_pos[a_id]
                    
                    manhattan_dist = abs(a_pos[0] - a_goal[0]) + abs(a_pos[1] - a_goal[1])
                    agent_reward = -0.1 * manhattan_dist
                    
                    if a_pos == a_goal:
                        agent_reward += 10.0
                    
                    if a_pos in self.obstacles_pos:
                        agent_reward -= 5.0
                    
                    team_reward += agent_reward
                
                rewards[agent_id] = team_reward / self.num_agents
                
            elif self.reward_type == 'sparse':
                # Only reward for reaching goal
                if current_pos == goal_pos:
                    rewards[agent_id] = 1.0
                else:
                    rewards[agent_id] = 0.0
        
        return rewards
    
    def _is_done(self, agent_id):
        """
        Check if an agent has reached its goal.
        
        Args:
            agent_id: ID of the agent to check
            
        Returns:
            bool: True if agent has reached its goal
        """
        return self.agent_positions[agent_id] == self.agent_goal_pos[agent_id]
    
    def _is_valid_position(self, x, y):
        """
        Check if a position is valid (within grid bounds and not an obstacle).
        
        Args:
            x, y: Position coordinates
            
        Returns:
            bool: True if position is valid
        """
        # Check if within grid bounds
        if not (0 <= x < self.grid_width and 0 <= y < self.grid_height):
            return False
        
        # Check if position has an obstacle
        if (x, y) in self.obstacles_pos:
            return False
        
        return True
        
    def render(self):
        """
        Render the environment as a simple text-based grid.
        """
        grid = [['.' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        
        # Mark obstacles
        for x, y in self.obstacles_pos:
            grid[y][x] = '#'
        
        # Mark goals
        for agent_id, (x, y) in self.agent_goal_pos.items():
            grid[y][x] = f'G{agent_id}'
        
        # Mark agents
        for agent_id, (x, y) in self.agent_positions.items():
            grid[y][x] = f'A{agent_id}'
        
        # Print the grid
        for row in grid:
            print(' '.join(row))
        print() 