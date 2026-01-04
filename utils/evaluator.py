import numpy as np
import torch

class Evaluator:
    """
    Evaluator class for testing the performance of MARL agents.
    Runs evaluation episodes in the environment and calculates average rewards.
    """
    def __init__(self, env, agent, device, policy_type, seed=None):
        """
        Initialize the Evaluator.

        Args:
            env: The environment instance (e.g., LBFWrapper).
            agent: The agent instance to be evaluated.
            device (str): Device to run the agent on ('cpu' or 'cuda').
            policy_type (str): Type of policy ('on-policy' or 'off-policy') to determine action selection method.
            seed (int, optional): Fixed seed for evaluation. If provided, it overrides the seed passed to evaluate().
        """
        self.env = env
        self.agent = agent
        self.device = device
        self.policy_type = policy_type 
        self.seed = seed
        
        env_info = self.env.get_env_info()
        self.n_agents = env_info['n_agents']
        
        print(f"[Evaluator] Initialized with policy type: {self.policy_type} | Seed: {self.seed}")

    def evaluate(self, n_episodes=5, seed=None):
        """
        Run evaluation episodes.

        Args:
            n_episodes (int): Number of episodes to evaluate.
            seed (int, optional): Base random seed for deterministic evaluation. 
                                  Ignored if self.seed was set in __init__.

        Returns:
            float: Mean episode reward across all agents and episodes.
        """
        self.agent.eval()
        rewards = []
        
        # Use self.seed if available, otherwise use the argument seed
        base_seed = self.seed if self.seed is not None else seed
        
        for i in range(n_episodes):
            # Set Seed for Deterministic Evaluation
            current_seed = base_seed + i if base_seed is not None else None
            
            # Reset Env
            obs, _ = self.env.reset(seed=current_seed)
            
            hidden = self.agent.init_hidden(1)
            
            episode_reward = 0
            done = False
            
            while not done:
                with torch.no_grad():
                    # Prepare Input: (1, N, ...)
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                    
                    # Take Action (Deterministic)
                    if self.policy_type == 'on-policy':
                        actions, _, _, next_hidden = self.agent.take_action(obs_tensor, hidden, evaluation=True)
                    else:
                        actions, next_hidden = self.agent.take_action(obs_tensor, hidden, current_step=0, evaluation=True)
                
                # Step Env
                # Handle tensor output to numpy list for env
                # Tensor -> Numpy
                if isinstance(actions, torch.Tensor):
                    actions = actions.cpu().numpy()
                
                if isinstance(actions, np.ndarray):
                    if actions.ndim > 1 and actions.shape[0] == 1:
                        action_input = actions[0].tolist()
                    else:
                        action_input = actions.tolist()
                else:
                    action_input = actions
                
                next_obs, reward, dones, truncated, _ = self.env.step(action_input)
                
                episode_reward += np.sum(reward)
                obs = next_obs
                hidden = next_hidden
                
                # Check termination
                if np.any(dones) or np.any(truncated):
                    done = True
            
            rewards.append(episode_reward)
        
        self.agent.train() # Switch back to train mode
        return np.mean(rewards)