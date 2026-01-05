import numpy as np
import torch

class ParallelReplayBuffer:
    """
    Parallel Replay Buffer for Off-Policy.
    Function: Experience replay buffer for Multi-Agent Reinforcement Learning (MARL) in parallel environments, especially suitable for Off-Policy algorithms.
    Role: Capable of storing transition data (obs, actions, rewards, etc.) from multiple parallel environments and supports sampling sequence data to train Recurrent Neural Networks (RNN).
    """
    def __init__(self, num_envs, env_info, capacity, batch_size, sequence_length, hidden_dim, rnn_layers=1, device='cpu'):
        """
        Initialize ParallelReplayBuffer.

        Parameters:
        - num_envs: Number of parallel environments.
        - env_info: Dictionary containing environment information (e.g., n_agents, obs_shape, state_shape).
        - capacity: Maximum time steps stored per environment (buffer capacity).
        - batch_size: Number of sequences per sample (Batch Size).
        - sequence_length: Length of sampled sequences (for RNN training).
        - hidden_dim: Dimension of RNN hidden layers.
        - rnn_layers: Number of RNN layers, default is 1.
        - device: Device for data storage and return (e.g., 'cpu' or 'cuda').
        """
        self.num_envs = num_envs
        self.n_agents = env_info["n_agents"]
        
        self.obs_shape = env_info["obs_shape"]
        self.obs_dim = self.obs_shape if isinstance(self.obs_shape, tuple) else (self.obs_shape,)

        self.state_shape = env_info.get("state_shape")
        self.state_dim = self.state_shape

        self.capacity = capacity
        self.batch_size = batch_size
        self.seq_len = sequence_length
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.device = device

        # === Storage ===
        # Obs: [Capacity, Num_Envs, N_Agents, ...]
        self.obs = np.zeros((capacity, num_envs, self.n_agents, *self.obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_envs, self.n_agents, *self.obs_dim), dtype=np.float32)
        
        self.state = np.zeros((capacity, num_envs, *self.state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, num_envs, *self.state_dim), dtype=np.float32)
        
        # Hidden: [Capacity, Num_Envs, N_Agents, RNN_Layers, Hidden_Dim]
        self.hidden = np.zeros((capacity, num_envs, self.n_agents, self.rnn_layers, self.hidden_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_envs, self.n_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, num_envs, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((capacity, num_envs, self.n_agents), dtype=np.float32)
        
        # Helper: Stores if ANY agent is done (Episode Done)
        self.global_dones = np.zeros((capacity, num_envs), dtype=bool)

        self.ptr = 0
        self.size = 0
        
        # Agent IDs Template: (1, 1, N_Agents)
        # Initialized on device to avoid transfer during sampling
        self.agent_ids_template = torch.arange(self.n_agents, device=device).reshape(1, 1, -1)

    def push(self, obs, hidden, state, actions, rewards, dones, next_obs, next_state):
        """
        Function: Store a new transition into the buffer.
        Role: Update data at the current pointer position and handle circular buffer overwriting.

        Inputs:
        - obs: Observation at the current time step. (numpy array, shape: (num_envs, n_agents, *obs_dim))
        - hidden: RNN hidden state at the current time step. (numpy array, shape: (num_envs, n_agents, rnn_layers, hidden_dim))
        - state: Global state at the current time step (optional). (numpy array, shape: (num_envs, *state_dim))
        - actions: Actions taken by agents. (numpy array, shape: (num_envs, n_agents))
        - rewards: Rewards received. (numpy array, shape: (num_envs, n_agents))
        - dones: Completion flags for agents. (numpy array, shape: (num_envs, n_agents))
        - next_obs: Observation at the next time step. (numpy array, shape: (num_envs, n_agents, *obs_dim))
        - next_state: Global state at the next time step (optional). (numpy array, shape: (num_envs, *state_dim))

        Outputs: None.
        """
        self.obs[self.ptr] = obs
        self.next_obs[self.ptr] = next_obs
        
        if state is not None: self.state[self.ptr] = state
        if next_state is not None: self.next_state[self.ptr] = next_state
            
        self.hidden[self.ptr] = hidden
        
        self.actions[self.ptr] = actions
        self.rewards[self.ptr] = rewards
        self.dones[self.ptr] = dones
        
        self.global_dones[self.ptr] = np.any(dones, axis=1)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self):
        """
        Function: Sample a batch of sequence data from the buffer.
        Role: Randomly select valid start indices and environment indices, construct sequence Batch for training, and handle Masking and Padding.

        Inputs: None.

        Outputs:
        - batch: A dictionary containing training data, structured as follows:
            - 'global': Contains global information.
                - 'mask': (torch.Tensor, shape: (batch_size, seq_len, 1))
                - 'dones': (torch.Tensor, shape: (batch_size, seq_len, 1))
                - 'state': (torch.Tensor, shape: (batch_size, seq_len, *state_dim)) [Optional]
                - 'next_state': (torch.Tensor, shape: (batch_size, seq_len, *state_dim)) [Optional]
            - 'all_agents': Contains information for all agents.
                - 'obs': (torch.Tensor, shape: (batch_size, seq_len, n_agents, *obs_dim))
                - 'next_obs': (torch.Tensor, shape: (batch_size, seq_len, n_agents, *obs_dim))
                - 'actions': (torch.Tensor, shape: (batch_size, seq_len, n_agents))
                - 'rewards': (torch.Tensor, shape: (batch_size, seq_len, n_agents))
                - 'dones': (torch.Tensor, shape: (batch_size, seq_len, n_agents))
                - 'agent_ids': (torch.Tensor, shape: (batch_size, seq_len, n_agents))
                - 'init_hidden': (torch.Tensor, shape: (rnn_layers, batch_size, n_agents, hidden_dim))
                - 'init_target_hidden': (torch.Tensor, shape: (rnn_layers, batch_size, n_agents, hidden_dim))
              Data format is typically Tensor, and on the specified device.
        """
        if self.size < self.seq_len:
            return None 

        valid_indices = []
        valid_envs = []
        needed = self.batch_size
        
        while len(valid_indices) < needed:
            remaining = needed - len(valid_indices)
            cand_time = np.random.randint(0, self.size, size=remaining)
            cand_env = np.random.randint(0, self.num_envs, size=remaining)
            
            is_contiguous = (cand_time + self.seq_len <= self.capacity)
            
            if self.size == self.capacity:
                # Danger zone: [ptr - seq_len, ptr] roughly speaking
                # Precise check: interval [cand, cand+seq] must NOT contain ptr
                crosses_ptr = (cand_time <= self.ptr) & (cand_time + self.seq_len > self.ptr)
                is_valid = is_contiguous & (~crosses_ptr)
            else:
                # Buffer not full: just don't go beyond ptr
                is_valid = (cand_time + self.seq_len <= self.ptr)
            
            if np.any(is_valid):
                valid_indices.extend(cand_time[is_valid])
                valid_envs.extend(cand_env[is_valid])
        
        # Select Indices
        idxs = np.array(valid_indices[:self.batch_size])
        next_idxs = (idxs + 1) % self.capacity
        env_idxs = np.array(valid_envs[:self.batch_size])
        
        # Time Indices: (B, T)
        seq_time_idxs = idxs[:, None] + np.arange(self.seq_len)[None, :]
        # Env Indices Broadcast: (B, T)
        seq_env_idxs = env_idxs[:, None].repeat(self.seq_len, axis=1)
        
        # Retrieve Data
        batch_obs = self.obs[seq_time_idxs, seq_env_idxs]
        batch_next_obs = self.next_obs[seq_time_idxs, seq_env_idxs]
        batch_actions = self.actions[seq_time_idxs, seq_env_idxs]
        batch_rewards = self.rewards[seq_time_idxs, seq_env_idxs]
        batch_dones = self.dones[seq_time_idxs, seq_env_idxs]
        
        raw_hidden = self.hidden[idxs, env_idxs]  # (B, N_Agents, RNN_Layers, H)
        batch_hidden = np.transpose(raw_hidden, (2, 0, 1, 3)) # (RNN_Layers, B, N_Agents, H)
        
        raw_next_hidden = self.hidden[next_idxs, env_idxs]
        batch_next_hidden = np.transpose(raw_next_hidden, (2, 0, 1, 3)) # (RNN_Layers, B, N_Agents, H)

        # Masking Logic
        batch_global_dones = self.global_dones[seq_time_idxs, seq_env_idxs] # (B, T)
        first_done_idx = np.argmax(batch_global_dones, axis=1)
        has_done = np.any(batch_global_dones, axis=1)
        valid_lens = np.where(has_done, first_done_idx + 1, self.seq_len)
        
        time_steps = np.arange(self.seq_len)[None, :]
        mask = (time_steps < valid_lens[:, None]).astype(np.float32)[:, :, None] # (B, T, 1)
        
        # Zero-Padding
        inv_mask = (1.0 - mask).astype(bool).squeeze(-1)
        batch_obs[inv_mask] = 0
        batch_next_obs[inv_mask] = 0
        batch_actions[inv_mask] = 0
        batch_rewards[inv_mask] = 0
        batch_dones[inv_mask] = 0
        
        # Agent IDs
        batch_agent_ids = self.agent_ids_template.expand(self.batch_size, self.seq_len, -1).to(self.device)

        batch = {
            'global': {
                'mask': torch.from_numpy(mask).to(self.device),
                'dones': torch.from_numpy(batch_global_dones.astype(np.float32)[:, :, None]).to(self.device)
            },
            'all_agents': {
                'obs': torch.from_numpy(batch_obs).to(self.device),
                'next_obs': torch.from_numpy(batch_next_obs).to(self.device),
                'actions': torch.from_numpy(batch_actions).to(self.device),
                'rewards': torch.from_numpy(batch_rewards).to(self.device),
                'dones': torch.from_numpy(batch_dones).to(self.device),
                'agent_ids': batch_agent_ids,
                'init_hidden': torch.from_numpy(batch_hidden).to(self.device),
                'init_target_hidden': torch.from_numpy(batch_next_hidden).to(self.device),
            }
        }

        batch_state = self.state[seq_time_idxs, seq_env_idxs]
        batch_next_state = self.next_state[seq_time_idxs, seq_env_idxs]
        batch_state[inv_mask] = 0
        batch_next_state[inv_mask] = 0
        
        batch['global']['state'] = torch.from_numpy(batch_state).to(self.device)
        batch['global']['next_state'] = torch.from_numpy(batch_next_state).to(self.device)

        return batch

    def __len__(self):
        return self.size