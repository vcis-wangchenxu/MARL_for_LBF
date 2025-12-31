import numpy as np
import torch

class ParallelRolloutBuffer:
    """
    Parallel Rollout Buffer for On-Policy MARL algorithms (e.g., MAPPO).
    Stores transitions from multiple parallel environments and supports RNN data chunking.

    Key Features:
    1. Stores observations, states, actions, rewards, etc., for multiple environments.
    2. Supports Recurrent Neural Networks (RNN) by storing hidden states.
    3. Provides a generator to yield data chunks for RNN training.
    """
    def __init__(self, num_envs, env_info, buffer_size, hidden_dim, rnn_layers=1, device='cpu'):
        """
        Initialize the buffer.

        Args:
            num_envs (int): Number of parallel environments.
            env_info (dict): Environment information (n_agents, obs_shape, state_shape, etc.).
            buffer_size (int): Length of the rollout (number of steps per environment).
            hidden_dim (int): Dimension of the RNN hidden state.
            rnn_layers (int): Number of RNN layers.
            device (str): Device to store tensors (used in generator).
        """
        self.num_envs = num_envs
        self.n_agents = env_info["n_agents"]
        
        # Handle tuple obs_shape
        self.obs_shape = env_info["obs_shape"]
        self.obs_dim = self.obs_shape if isinstance(self.obs_shape, tuple) else (self.obs_shape,)

        self.state_shape = env_info.get("state_shape", None)
        self.use_state = self.state_shape is not None and self.state_shape != 0
        self.state_dim = self.state_shape if self.use_state else (0,)

        self.buffer_size = buffer_size    # Rollout length
        self.rnn_layers = rnn_layers
        self.hidden_dim = hidden_dim
        self.device = device

        # === Storage ===
        self.obs = np.zeros((buffer_size + 1, num_envs, self.n_agents, *self.obs_dim), dtype=np.float32)    # +1 for last obs
        
        if self.use_state:
            self.state = np.zeros((buffer_size + 1, num_envs, *self.state_dim), dtype=np.float32)
        else:
            self.state = None
        
        self.hidden_states = np.zeros((buffer_size + 1, num_envs, self.n_agents, self.rnn_layers, self.hidden_dim), dtype=np.float32)

        # Assuming Discrete Actions for int64
        self.actions = np.zeros((buffer_size, num_envs, self.n_agents), dtype=np.int64) 
        self.rewards = np.zeros((buffer_size, num_envs, self.n_agents), dtype=np.float32)
        self.dones = np.zeros((buffer_size + 1, num_envs, self.n_agents), dtype=np.float32)
        
        self.log_probs = np.zeros((buffer_size, num_envs, self.n_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size + 1, num_envs, self.n_agents), dtype=np.float32)

        # Agent IDs: Pre-calculate a small batch template
        # Shape must be (1, 1, N_Agents) to expand to (Batch, Time, N_Agents)
        self.agent_ids_template = torch.arange(self.n_agents, device=device).reshape(1, 1, -1)

        self.step = 0
        
    def is_full(self):
        """
        Check if the buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return self.step >= self.buffer_size

    def push(self, obs, state, hidden_states, actions, rewards, dones, log_probs, values):
        """
        Store a transition step.

        Args:
            obs (np.ndarray): Observations. Shape: (num_envs, n_agents, *obs_shape).
            state (np.ndarray): Global states. Shape: (num_envs, *state_shape).
            hidden_states (np.ndarray): RNN hidden states. Shape: (num_envs, n_agents, rnn_layers, hidden_dim).
            actions (np.ndarray): Actions taken. Shape: (num_envs, n_agents).
            rewards (np.ndarray): Rewards received. Shape: (num_envs, n_agents).
            dones (np.ndarray): Done flags. Shape: (num_envs, n_agents).
            log_probs (np.ndarray): Log probabilities of actions. Shape: (num_envs, n_agents).
            values (np.ndarray): Value estimates. Shape: (num_envs, n_agents).
        """
        if self.step >= self.buffer_size:
            raise IndexError("Rollout Buffer is full!")

        self.obs[self.step] = obs
        if self.use_state and state is not None:
            self.state[self.step] = state
            
        self.hidden_states[self.step] = hidden_states 
        self.actions[self.step] = actions
        self.rewards[self.step] = rewards
        self.dones[self.step] = dones
        self.log_probs[self.step] = log_probs
        self.values[self.step] = values
        
        self.step += 1

    def insert_last_step(self, obs, state, hidden_states, values, dones):
        """
        Store the data for the last step (t=T+1), required for GAE calculation.

        Args:
            obs (np.ndarray): Observations at T+1. Shape: (num_envs, n_agents, *obs_shape).
            state (np.ndarray): Global states at T+1. Shape: (num_envs, *state_shape).
            hidden_states (np.ndarray): Hidden states at T+1. Shape: (num_envs, n_agents, rnn_layers, hidden_dim).
            values (np.ndarray): Value estimates at T+1. Shape: (num_envs, n_agents).
            dones (np.ndarray): Done flags at T+1. Shape: (num_envs, n_agents).
        """
        self.obs[self.step] = obs
        if self.use_state and state is not None:
            self.state[self.step] = state
        self.hidden_states[self.step] = hidden_states
        self.values[self.step] = values
        self.dones[self.step] = dones

    def get_data(self):
        """
        Retrieve all stored data for GAE calculation.

        Returns:
            data (dict): Dictionary containing numpy arrays of all stored data.
                - obs: (T, num_envs, n_agents, ...)
                - actions: (T, num_envs, n_agents)
                - ...
        """
        T = self.buffer_size
        
        data = {
            'obs': self.obs[:T],
            'hidden_states': self.hidden_states[:T],
            'actions': self.actions[:T],
            'rewards': self.rewards[:T],
            'dones': self.dones[:T],
            'log_probs': self.log_probs[:T],
            'values': self.values[:T + 1],
            'masks': 1.0 - self.dones[:T], 
        }
        if self.use_state:
            data['state'] = self.state[:T]
            
        return data

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        """
        Generator that yields training batches with RNN data chunking.
        
        Args:
            advantages (np.ndarray): Calculated advantages. Shape: (T, num_envs, n_agents).
            num_mini_batch (int): Number of mini-batches per epoch.
            data_chunk_length (int): Length of time chunks for RNN training.

        Yields:
            tuple: A tuple containing tensors for training:
                - obs_batch: (Batch_Size, Chunk_Len, n_agents, *obs_shape)
                - hidden_states_batch: (Batch_Size, Layers, n_agents, hidden_dim)
                - actions_batch: (Batch_Size, Chunk_Len, n_agents)
                - values_batch: (Batch_Size, Chunk_Len, n_agents)
                - returns_batch: (Batch_Size, Chunk_Len, n_agents)
                - log_probs_batch: (Batch_Size, Chunk_Len, n_agents)
                - advantages_batch: (Batch_Size, Chunk_Len, n_agents)
                - masks_batch: (Batch_Size, Chunk_Len, n_agents)
                - agent_ids: (Batch_Size, Chunk_Len, n_agents)
        """
        T = self.buffer_size # Rollout length
        assert T % data_chunk_length == 0, "Buffer size must be divisible by chunk length"

        num_chunks_per_env = T // data_chunk_length          # Chunks per environment
        total_chunks = num_chunks_per_env * self.num_envs    # Total chunks across all envs
        
        def _reshape_to_chunks(x):    # x: (T, Envs, Agents, ...)
            s = x.shape
            # Split T into (Num_Chunks, Chunk_Len)
            x = x[:num_chunks_per_env * data_chunk_length]
            x_reshaped = x.reshape(num_chunks_per_env, data_chunk_length, self.num_envs, self.n_agents, *s[3:])
            
            # Transpose to merge Num_Chunks and Num_Envs
            # Target: (Num_Chunks, Num_Envs, Chunk_Len, N_Agents, ...)
            x_permuted = x_reshaped.transpose(0, 2, 1, 3, *range(4, len(x_reshaped.shape)))
            
            # Flatten to (Batch_Size, Chunk_Len, N_Agents, ...)
            # Batch_Size = Total_Chunks = Num_Chunks * Num_Envs
            return x_permuted.reshape(total_chunks, data_chunk_length, self.n_agents, *s[3:])

        # Batch Processing
        batch_obs = _reshape_to_chunks(self.obs[:T])
        batch_actions = _reshape_to_chunks(self.actions[:T])
        batch_log_probs = _reshape_to_chunks(self.log_probs[:T])
        batch_values = _reshape_to_chunks(self.values[:T])
        batch_returns = _reshape_to_chunks(advantages + self.values[:T])
        batch_dones = _reshape_to_chunks(self.dones[:T])
        batch_advantages = _reshape_to_chunks(advantages)
        
        # Calculate Masks on the fly
        batch_masks = 1.0 - batch_dones

        # Hidden States: take start of each chunk
        chunk_indices = np.arange(0, T, data_chunk_length)
        start_hidden = self.hidden_states[chunk_indices] # (Num_Chunks, Envs, Agents, H)
        
        batch_hidden_raw = start_hidden.reshape(total_chunks, self.n_agents, self.rnn_layers, self.hidden_dim) # (Total_Chunks, N_Agents, Layers, H)


        indices = np.arange(total_chunks)
        np.random.shuffle(indices)
        
        mini_batch_size = total_chunks // num_mini_batch

        for i in range(0, total_chunks, mini_batch_size):
            mb_indices = indices[i : i + mini_batch_size]
            
            # Create Agent IDs on the fly
            # Shape: (Batch_Size, Chunk_Len, N_Agents)
            current_batch_size = len(mb_indices)
            
            # Now expands correctly from (1, 1, N) to (B, T, N)
            batch_agent_ids = self.agent_ids_template.expand(current_batch_size, data_chunk_length, -1).to(self.device)

            mb_hidden = batch_hidden_raw[mb_indices] # (Batch_Size, N_Agents, Layers, H)
            mb_hidden = mb_hidden.transpose(2, 0, 1, 3) # (Layers, Batch_Size, N_Agents, H)

            yield (
                torch.from_numpy(batch_obs[mb_indices]).to(self.device),
                torch.from_numpy(mb_hidden).to(self.device),
                torch.from_numpy(batch_actions[mb_indices]).to(self.device),
                torch.from_numpy(batch_values[mb_indices]).to(self.device),
                torch.from_numpy(batch_returns[mb_indices]).to(self.device),
                torch.from_numpy(batch_log_probs[mb_indices]).to(self.device),
                torch.from_numpy(batch_advantages[mb_indices]).to(self.device),
                torch.from_numpy(batch_masks[mb_indices]).to(self.device),
                batch_agent_ids 
            )

    def clear(self):
        """
        Reset the buffer index to 0.
        """
        self.step = 0