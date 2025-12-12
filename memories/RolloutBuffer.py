from typing import Dict, List, Any
import numpy as np

# ====================================================================================
# 高效 On-Policy 滚动缓冲区 (Efficient Rollout Buffer)
# ------------------------------------------------------------------------------------
# 适用算法：
# 1. MAPPO: 需要 global_state (Critic) 和 obs (Actor)。
# 2. IPPO:  通常只需要 obs。本 Buffer 支持存入 state，如果不需要可传占位符或忽略。
# 3. PPO:   单智能体或多智能体通用。
#
# 特性：
# - 预分配 NumPy 数组，避免动态扩容开销。
# - 显式分离 全局数据 (State, Global Dones) 和 个体数据 (Obs, Actions, LogProbs)。
# ====================================================================================

class OnPolicyRolloutBuffer:
    """
    高效 On-Policy Buffer，支持存储 Global State 以适配 MAPPO/CTDE 架构。
    """
    def __init__(self, env_info: dict, buffer_size: int = 1000, device='cpu'):
        """
        :param env_info: 环境信息，需包含 n_agents, obs_shape, n_actions, state_shape
        :param buffer_size: 缓冲区容量 (对应 PPO 的 rollout_length / num_steps)
        :param device: 设备占位符
        """
        self.n_agents = env_info['n_agents']
        
        # --- 观测空间 (Actor 输入) ---
        self.obs_shape = env_info['obs_shape']
        # 确保是 tuple 格式
        self.obs_dim = self.obs_shape if isinstance(self.obs_shape, tuple) else (self.obs_shape,)
        
        # --- 全局状态空间 (Critic 输入 - MAPPO 关键) ---
        # 如果 env_info 中没有 state_shape，默认为 (0,) (即不存储有效状态)
        self.state_shape = env_info.get('state_shape', (0,))
        self.state_dim = self.state_shape if isinstance(self.state_shape, tuple) else (self.state_shape,)

        self.buffer_size = buffer_size
        self.agent_ids = list(range(self.n_agents))
        self.device = device

        # === 1. 预分配全局数据 (Shared Data) ===
        # Global State: (Capacity, State_Dim) -> MAPPO Critic 使用
        self.global_state = np.zeros((buffer_size, *self.state_dim), dtype=np.float32)
        self.next_global_state = np.zeros((buffer_size, *self.state_dim), dtype=np.float32)
        
        # Global Dones & Rewards
        self.dones_global = np.zeros(buffer_size, dtype=np.float32)
        self.collective_rewards = np.zeros(buffer_size, dtype=np.float32)

        # === 2. 预分配个体数据 (Agent-Specific Data) ===
        # 结构: dict[agent_id] -> { "obs": array, "actions": array ... }
        # 每个 Array 的第一维都是 buffer_size
        self.agent_buffers = {}
        for agent_id in self.agent_ids:
            self.agent_buffers[agent_id] = {
                # Actor 输入
                "obs": np.zeros((buffer_size, *self.obs_dim), dtype=np.float32),
                "next_obs": np.zeros((buffer_size, *self.obs_dim), dtype=np.float32),
                
                # 动作与策略信息
                "actions": np.zeros(buffer_size, dtype=np.int64), # 离散动作
                "log_probs": np.zeros(buffer_size, dtype=np.float32),
                
                # 价值信息 (用于计算 Advantage)
                "values": np.zeros(buffer_size, dtype=np.float32),
                "rewards": np.zeros(buffer_size, dtype=np.float32),
                "dones": np.zeros(buffer_size, dtype=np.float32),
            }

        # 写入指针
        self.ptr = 0

    def push(self, 
             obs: np.ndarray,           # (n_agents, *obs_dim)
             state: np.ndarray,         # (state_dim,)  <-- MAPPO 需要的全局状态
             actions: np.ndarray,       # (n_agents,)
             rewards: np.ndarray,       # (n_agents,)
             dones: np.ndarray,         # (n_agents,)
             log_probs: np.ndarray,     # (n_agents,)
             values: np.ndarray,        # (n_agents,)
             next_obs: np.ndarray,      # (n_agents, *obs_dim)
             next_state: np.ndarray,    # (state_dim,)  <-- MAPPO 需要的全局下一状态
             info: Dict[str, Any] = {}):        
        """
        存入一步数据。
        注意：传入的 state 应该是拼接好或处理好的全局特征向量/图像。
        """
        if self.ptr >= self.buffer_size:
            raise IndexError("RolloutBuffer is full! Call clear() or update() before pushing.")

        # --- 1. 存入全局数据 ---
        self.global_state[self.ptr] = state
        self.next_global_state[self.ptr] = next_state
        
        # 只要有一个智能体 Done，通常视为 Global Done (用于计算全局 Returns)
        self.dones_global[self.ptr] = np.any(dones)
        
        # 存储集体奖励 (用于 Cooperative 任务评估)
        collective_reward = info.get('collective_reward', np.sum(rewards))
        self.collective_rewards[self.ptr] = collective_reward

        # --- 2. 存入个体数据 ---
        for i in self.agent_ids:
            buf = self.agent_buffers[i]
            
            buf["obs"][self.ptr] = obs[i]
            buf["next_obs"][self.ptr] = next_obs[i]
            buf["actions"][self.ptr] = actions[i]
            buf["log_probs"][self.ptr] = log_probs[i]
            buf["values"][self.ptr] = values[i]
            buf["rewards"][self.ptr] = rewards[i]
            buf["dones"][self.ptr] = dones[i]

        self.ptr += 1

    def get_all_data(self) -> Dict[str, Any]:
        """
        取出当前缓冲区内的所有有效数据 (0 到 ptr)。
        返回 Numpy 数组，后续可转换为 Tensor 供 PPO 更新使用。
        """
        T = self.ptr
        
        # 全局数据
        data = {
            "global_state": self.global_state[:T],          # (T, State_Dim)
            "next_global_state": self.next_global_state[:T],# (T, State_Dim)
            "dones": self.dones_global[:T],                 # (T,)
            "collective_rewards": self.collective_rewards[:T],
        }
        
        # 个体数据
        for i in self.agent_ids:
            buf = self.agent_buffers[i]
            data[i] = {
                "obs": buf["obs"][:T],             # (T, *Obs_Dim)
                "next_obs": buf["next_obs"][:T],   # (T, *Obs_Dim)
                "actions": buf["actions"][:T],     # (T,)
                "log_probs": buf["log_probs"][:T], # (T,)
                "values": buf["values"][:T],       # (T,)
                "rewards": buf["rewards"][:T],     # (T,)
                "dones": buf["dones"][:T],         # (T,)
            }
            
        return data
    
    def clear(self) -> None:
        """
        清空缓冲区 (重置指针)
        """
        self.ptr = 0
    
    def __len__(self) -> int:
        return self.ptr