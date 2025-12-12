import gymnasium as gym
import numpy as np
import lbforaging

class LBFWrapper(gym.Wrapper):
    def __init__(self, env_id, **kwargs):
        kwargs["grid_observation"] = True 
        kwargs["observe_agent_levels"] = True

        # 关闭 checker 消除警告
        env = gym.make(env_id, disable_env_checker=True, **kwargs) 
        super().__init__(env)

        self.n_agents = self.env.unwrapped.n_agents
        self.obs_shape = env.observation_space[0].shape
        self.n_actions = env.action_space[0].n
        self.state_shape = (self.n_agents * self.obs_shape[0], *self.obs_shape[1:])
        self.max_episode_steps = getattr(env.unwrapped, "_max_episode_steps", 50)
    
    def get_env_info(self):
        info = {
            "n_agents": self.n_agents,
            "obs_shape": self.obs_shape,
            "n_actions": self.n_actions,
            "state_shape": self.state_shape,
            "max_episode_steps": self.max_episode_steps,
        }
        return info
        
    def reset(self, seed=None):
        obss, info = self.env.reset(seed=seed)
        # 将 Tuple 转为 Numpy Array: [n_agents, obs_dim]
        return np.array(obss), info

    def step(self, actions):
        # actions 是 list 或 array
        next_obss, rewards, done, truncated, info = self.env.step(actions)
        
        # 标准化返回值
        next_obss = np.array(next_obss) # [n_agents, obs_dim]
        rewards = np.array(rewards)     # [n_agents]
        # v3版本 done 是单个 bool，把它扩展成 [n_agents] 的数组方便计算
        dones = np.array([done or truncated] * self.n_agents) 
        
        return next_obss, rewards, dones, info