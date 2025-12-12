from collections import deque
from typing import Dict, List, Any, Union
import numpy as np

class OffPolicyReplayBuffer:
    def __init__(self, env_info, capacity, batch_size, sequence_length, device='cpu'):
        """
        初始化 Replay Buffer
        :param env_info: 环境信息字典，需包含 n_agents, obs_shape, n_actions
        :param capacity: 缓冲区最大容量
        :param batch_size: 采样 Batch 大小
        :param sequence_length: 采样的序列长度 (DRQN 需要)
        :param device: 设备 (保留参数，本类只负责 Numpy 存储)
        """
        self.n_agents = env_info["n_agents"]
        self.obs_shape = env_info["obs_shape"]
        self.n_actions = env_info["n_actions"]
        # 处理 obs_shape 可能是 tuple 的情况
        self.obs_dim = env_info["obs_shape"] if isinstance(env_info["obs_shape"], tuple) else (env_info["obs_shape"],)

        self.state_shape = env_info.get("state_shape", None)
        if self.state_shape is None:
            self.state_dim = (0,)
        elif isinstance(self.state_shape, int):
            self.state_dim = (self.state_shape,)
        else:
            self.state_dim = self.state_shape

        self.capacity = capacity
        self.batch_size = batch_size
        self.seq_len = sequence_length
        self.device = device

        # === 预分配大数组 (Pre-allocated Memory) ===
        # obs 存储格式: [Capacity, N_Agents, C, H, W]
        self.obs = np.zeros((capacity, self.n_agents, *self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, self.n_agents, *self.obs_dim), dtype=np.float32)
        
        self.state = np.zeros((capacity, *self.state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, *self.state_dim), dtype=np.float32)
        
        self.actions = np.zeros((capacity, self.n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((capacity, self.n_agents), dtype=np.float32)

        # 辅助变量
        self.ptr = 0        # 当前写入位置的指针 (Write Head)
        self.size = 0       # 当前缓冲区已存储的数据量
        
        # 记录每一步是否是 Episode 的结束
        # 用于在采样时判断序列是否中断
        self.episode_dones = np.zeros(capacity, dtype=bool)

    def push(self, obs, state, actions, rewards, dones, next_obs, next_state, info):
        """
        存入一步数据 (Push one transition)
        逻辑：
        1. 直接将 Numpy 数组写入当前 self.ptr 指向的内存位置，无需 append。
        2. 如果缓冲区已满，ptr 会回到开头 (Ring Buffer)，覆盖最旧的数据。
        """
        self.obs[self.ptr] = obs                 # (N, C, H, W)
        self.next_obs[self.ptr] = next_obs       # (N, C, H, W)
        self.actions[self.ptr] = actions         # (N, )
        self.rewards[self.ptr] = rewards         # (N, )
        self.dones[self.ptr] = dones             # (N, )
        
        self.state[self.ptr] = state              # (State_Dim, )
        self.next_state[self.ptr] = next_state    # (State_Dim, )
        
        global_done = np.any(dones)
        self.episode_dones[self.ptr] = global_done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        """
        采样一个 Batch 的序列数据 (Sample with Zero-Padding)
        优化：使用 NumPy 向量化操作替代 Python 循环生成 Mask，提升采样速度。
        """
        
        # --- 1. 随机寻找合法的起始索引 ---
        valid_indices = []
        timeout = 0
        max_idx = self.capacity - self.seq_len if self.size == self.capacity else self.size - self.seq_len

        if max_idx <= 0:
            raise ValueError("Buffer size is too small for the requested sequence length!")

        while len(valid_indices) < self.batch_size:
            idx = np.random.randint(0, max_idx)
            # 写指针保护：防止读取到正在被覆盖的数据
            if self.ptr > idx and self.ptr < idx + self.seq_len:
                continue
            valid_indices.append(idx)
            
            timeout += 1
            if timeout > 100000:
                raise RuntimeError("Cannot find valid sequences! Check buffer capacity.")

        idxs = np.array(valid_indices) # (B, )
        
        # --- 2. 向量化取出数据 ---
        # 生成索引矩阵 (B, T)
        seq_idxs = idxs[:, None] + np.arange(self.seq_len)[None, :]
        seq_idxs = seq_idxs % self.capacity

        # 一次性取出所有数据
        batch_obs = self.obs[seq_idxs]           # (B, T, N, ...)
        batch_next_obs = self.next_obs[seq_idxs] # (B, T, N, ...)
        batch_actions = self.actions[seq_idxs]   # (B, T, N)
        batch_rewards = self.rewards[seq_idxs]   # (B, T, N)
        batch_dones = self.dones[seq_idxs]       # (B, T, N)

        batch_state = self.state[seq_idxs]           # (B, T, State_Dim)
        batch_next_state = self.next_state[seq_idxs] # (B, T, State_Dim)

        batch_global_dones = self.episode_dones[seq_idxs] # (B, T)

        # --- 3. 向量化处理 Mask 和 Zero-Padding (优化核心) ---
        
        # 计算每个序列中第一个 done 的位置
        # argmax 会返回第一个 True 的索引；如果全为 False，则返回 0
        first_done_indices = np.argmax(batch_global_dones, axis=1)
        
        # 检查序列中是否真的包含 done (因为 argmax 在全 False 时也返回 0，需要区分)
        has_dones = np.any(batch_global_dones, axis=1)
        
        # 如果没有 done，有效长度就是 seq_len；如果有 done，有效长度就是 done 的索引 + 1 (包含 done 这一帧)
        valid_lengths = np.where(has_dones, first_done_indices + 1, self.seq_len)
        
        # 生成 Mask: shape (B, T)
        # 利用广播: [0, 1, 2... T-1] < [len_0, len_1, ... len_B]
        time_indices = np.arange(self.seq_len)[None, :]
        mask_bool = time_indices < valid_lengths[:, None]
        mask = mask_bool.astype(np.float32)[:, :, None] # (B, T, 1)

        # 执行 Zero-Padding (将无效部分置零)
        # 利用 mask 的广播特性，无效部分乘以 0 即可
        # 注意：这里需要扩展 mask 维度以匹配数据维度，或者直接利用 boolean indexing 赋值
        # 为简单起见，直接用 mask 相乘 (假设 mask 为 0 的地方数据也该是 0)
        # 但严谨的做法是把 mask=0 的位置的数据设为 0
        
        # 简单高效的置零方法：
        inverted_mask = ~mask_bool # (B, T) 无效位置为 True
        
        batch_obs[inverted_mask] = 0
        batch_next_obs[inverted_mask] = 0
        batch_actions[inverted_mask] = 0
        batch_rewards[inverted_mask] = 0
        batch_dones[inverted_mask] = 0

        batch_state[inverted_mask] = 0
        batch_next_state[inverted_mask] = 0

        # --- 4. 组装返回 ---
        batch = {}
        batch['global'] = {
            'mask': mask,           # (B, T, 1)
            'state': batch_state,   # (B, T, State_Dim)
            'next_state': batch_next_state, # (B, T, State_Dim)
        } 

        for i in range(self.n_agents):
            batch[i] = {
                'obs': batch_obs[:, :, i],
                'next_obs': batch_next_obs[:, :, i],
                'actions': batch_actions[:, :, i],   
                'rewards': batch_rewards[:, :, i],
                'dones': batch_dones[:, :, i],
            }
            
        return batch

    def __len__(self):
        return self.size