import gymnasium as gym
import numpy as np
import lbforaging

class LBFWrapper(gym.Wrapper):
    """
    LBFWrapper class wraps the Level-Based Foraging (LBF) environment to adapt it to the interface of multi-agent reinforcement learning algorithms.
    Main functions include:
    1. Enforce grid observation (grid_observation) and include agent levels (observe_agent_levels).
    2. Construct global state shape (state_shape).
    3. Unify observation space (observation_space) to (n_agents, C, H, W).
    4. Standardize the format and dimensions of return values in the step method.
    """
    def __init__(self, env_id, **kwargs):
        """
        Initialize LBFWrapper.

        Args:
            env_id (str): Environment ID, e.g., "Foraging-8x8-2p-2f-v3".
            **kwargs: Other arguments passed to gym.make.
        """
        kwargs["grid_observation"] = True           # Use grid observation
        kwargs["observe_agent_levels"] = True       # Include agent levels in observations

        # Disable checker to suppress warnings
        env = gym.make(env_id, disable_env_checker=True, **kwargs) 
        super().__init__(env)  # <-- self.env

        self.n_agents = self.env.unwrapped.n_agents
        self.obs_shape = env.observation_space[0].shape
        self.n_actions = env.action_space[0].n
        
        # Construct global State Shape 
        # (N, C, H, W) -> (N*C, H, W)
        self.state_shape = (self.n_agents * self.obs_shape[0], *self.obs_shape[1:])
        self.max_episode_steps = getattr(env.unwrapped, "_max_episode_steps", 50)
        
        single_agent_obs_space = env.observation_space[0]

        # single_obs.low is (C, H, W) -> expand to (N, C, H, W)
        low = np.tile(single_agent_obs_space.low[None, ...], (self.n_agents, 1, 1, 1))
        high = np.tile(single_agent_obs_space.high[None, ...], (self.n_agents, 1, 1, 1))
        
        self.observation_space = gym.spaces.Box(
            low=low, 
            high=high, 
            dtype=single_agent_obs_space.dtype
        )
        
    def get_env_info(self):
        """
        Get basic environment information.

        Returns:
            info (dict): Dictionary containing environment information.
                - n_agents (int): Number of agents.
                - obs_shape (tuple): Observation shape of a single agent (C, H, W).
                - n_actions (int): Size of action space.
                - state_shape (tuple): Global state shape (N*C, H, W).
                - max_episode_steps (int): Maximum number of steps.
        """
        info = {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_shape,                   # Observation shape of a single agent (C, H, W)
            "n_actions": self.n_actions,
            "state_shape": self.state_shape,              # Global state shape (N*C, H, W)
            "max_episode_steps": self.max_episode_steps,
        }
        return info
        
    def reset(self, seed=None, options=None):
        """
        Reset the environment.

        Args:
            seed (int, optional): Random seed.
            options (dict, optional): Additional options.

        Returns:
            obss (np.ndarray): Initial observations of all agents. Shape: (n_agents, C, H, W).
            info (dict): Additional information.
        """
        # Gymnasium API requires reset to accept seed and options.
        obss, info = self.env.reset(seed=seed, options=options)

        return np.array(obss), info     # Rturn shape: (n_agents, C, H, W)

    def step(self, actions):
        """
        Execute one step of action.

        Args:
            actions (list or np.ndarray): List or array of actions for all agents. Shape: (n_agents,).

        Returns:
            next_obss (np.ndarray): Next observations for all agents. Shape: (n_agents, C, H, W).
            rewards (np.ndarray): Rewards for all agents. Shape: (n_agents,).
            dones (np.ndarray): Done flags for all agents. Shape: (n_agents,).
            truncated (bool): Whether truncated.
            info (dict): Additional information.
        """
        # actions is a list or array
        next_obss, rewards, done, truncated, info = self.env.step(actions)

        # Standardize return values
        next_obss = np.array(next_obss) # [n_agents, obs_dim]
        rewards = np.array(rewards)     # [n_agents]
        # In v3 version, done is a single bool, expand it to an array of [n_agents] for easier calculation
        dones = np.array([done or truncated] * self.n_agents)  # [n_agents]
        truncateds = np.array([truncated] * self.n_agents)     # [n_agents]
        
        return next_obss, rewards, dones, truncateds, info
    
if __name__ == "__main__":
    env = LBFWrapper("Foraging-8x8-2p-2f-v3")
    obs, info = env.reset()
    print("Obs shape:", obs.shape)  # Should be (n_agents, C, H, W)
    print("Env info:", env.get_env_info())