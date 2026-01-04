from typing import Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.DRQN import AgentDRQN

class VDN:
    """
    Value Decomposition Networks (VDN) algorithm.
    Core idea: Q_tot(s, a) = Sum(Q_i(s^i, a^i))
    Function: Implements Value Decomposition Networks (VDN) algorithm for multi-agent cooperative tasks.
    Role: Decomposes global Q-value into sum of local Q-values of agents, training agents by maximizing global Q-value. Supports parameter sharing or independent networks.
    """
    def __init__(self, env_info: dict, args: Dict):
        """
        Initialize VDN Algorithm.

        Parameters:
        - env_info: Environment information dictionary, containing:
            - n_agents: Number of agents
            - n_actions: Size of action space
            - obs_shape: Shape of observation space
        - args: Arguments object containing hyperparameter configurations (e.g., lr, gamma, hidden_dim, norm_factor, etc.).
        """
        self.env_info = env_info
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        self.obs_shape = env_info['obs_shape']

        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq

        # Get hyperparameters 
        self.hidden_dim = args.rnn_hidden_dim
        self.norm_factor = args.norm_factor
        self.tau = args.target_tau

        self.share_params = args.share_parameters

        # Epsilon-Greedy parameters
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.epsilon = self.epsilon_start
        
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cpu')

        if self.share_params:
            print(f"[VDN] Parameter Sharing Enabled. All {self.n_agents} agents share one DRQN.")
            self.agent = AgentDRQN(self.obs_shape, self.n_actions, 
                                   self.n_agents, self.hidden_dim,  
                                   norm_factor=self.norm_factor, 
                                   use_agent_id=True).to(self.device)

            self.target_agent = AgentDRQN(self.obs_shape, self.n_actions, 
                                          self.n_agents, self.hidden_dim, 
                                          norm_factor=self.norm_factor, 
                                          use_agent_id=True).to(self.device)
            self.target_agent.load_state_dict(self.agent.state_dict())
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr)

        else:
            print(f"[VDN] Parameter Sharing Disabled. Creating {self.n_agents} independent DRQNs.")
            # Note: Independent VDN usually doesn't need Agent ID if they have separate networks.
            self.agents = nn.ModuleList([
                AgentDRQN(self.obs_shape, self.n_actions, 
                          1, self.hidden_dim, # n_agents=1 for independent net view if desired, or n_agents
                          norm_factor=self.norm_factor, 
                          use_agent_id=False) 
                for _ in range(self.n_agents)
            ]).to(self.device)
            
            self.target_agents = nn.ModuleList([
                AgentDRQN(self.obs_shape, self.n_actions, 
                          1, self.hidden_dim, 
                          norm_factor=self.norm_factor, 
                          use_agent_id=False) 
                for _ in range(self.n_agents)
            ]).to(self.device)
            self.target_agents.load_state_dict(self.agents.state_dict())
            self.optimizer = optim.Adam(self.agents.parameters(), lr=self.lr)

        self.criterion = nn.MSELoss()
        self._train_step_count = 0

    def init_hidden(self, batch_size=1) -> torch.Tensor:
        """
        Initialize RNN hidden states.
        Returns tensor of shape: (Batch, N_Agents, Layers, Hidden)
        This matches the structure expected by ReplayBuffer and the loop in train_offpolicy.
        Function: Initialize RNN hidden states.
        Role: Generate zero-filled hidden states for a new episode or batch.

        Inputs:
        - batch_size: Batch size (int, default 1).

        Outputs:
        - hidden_state: Initialized hidden state.
          Type: torch.Tensor
          Shape: (Batch, N_Agents, Layers, Hidden)
        """
        if self.share_params:
            # Init for all agents in all envs at once
            # Agent init_hidden returns (Layers, Total_Batch, Hidden)
            # We treat Total_Batch = batch_size * n_agents
            h = self.agent.init_hidden(batch_size * self.n_agents, device=self.device) # (L, B*N, H)
            # Reshape to (L, B, N, H)
            h = h.view(self.agent.rnn_layers, batch_size, self.n_agents, self.hidden_dim)
            # Permute to (B, N, L, H) for storage/passing
            h = h.permute(1, 2, 0, 3).contiguous()
            return h
        else:
            # List of (L, B, H)
            h_list = [agent.init_hidden(batch_size, device=self.device) for agent in self.agents]
            # Stack to (N, L, B, H) -> (L, B, N, H) (if we permute)
            # Let's standardize to (B, N, L, H)
            h_stack = torch.stack(h_list, dim=0) # (N, L, B, H)
            h = h_stack.permute(2, 0, 1, 3).contiguous() # (B, N, L, H)
            return h

    @torch.no_grad()
    def take_action(self, obs_tensor, hidden_state, current_step, evaluation=False) -> Tuple[np.ndarray, torch.Tensor]:
        """
        obs_tensor: (B, N, C, H, W)
        hidden_state: (B, N, Layers, Hidden)
        Function: Select actions based on observations and hidden states.
        Role: Forward pass through DRQN network, select action using Epsilon-Greedy strategy, and return new hidden state.

        Inputs:
        - obs_tensor: Observation data.
          Type: torch.Tensor
          Shape: (Batch, N_Agents, *obs_shape) (e.g., (B, N, C, H, W)) [B <- num_envs]
        - hidden_state: RNN hidden state.
          Type: torch.Tensor
          Shape: (Batch, N_Agents, Layers, Hidden)
        - current_step: Current training step (for epsilon decay).
        - evaluation: Whether in evaluation mode (bool). If True, epsilon is 0; otherwise use decayed epsilon.

        Outputs:
        - final_actions: Selected actions.
          Type: np.ndarray
          Shape: (Batch, N_Agents)
        - next_hidden_state: Updated hidden state.
          Type: torch.Tensor
          Shape: (Batch, N_Agents, Layers, Hidden)
        """
        batch_size = obs_tensor.shape[0]

        # Update Epsilon
        if evaluation:
            self.epsilon = 0.0
            explore_mask = np.zeros((batch_size, self.n_agents), dtype=bool)
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon_start - \
                           (self.epsilon_start - self.epsilon_end) * (current_step / self.epsilon_decay))
            rand_probs = np.random.rand(batch_size, self.n_agents)
            explore_mask = rand_probs < self.epsilon

        if self.share_params:
            # Flatten Obs: (B, N, C, H, W) -> (B*N, C, H, W)
            B, N, C, H, W = obs_tensor.shape
            obs_flat = obs_tensor.view(B * N, C, H, W)
            
            # Prepare Hidden: (B, N, L, H) -> (L, B*N, H)
            h_perm = hidden_state.permute(2, 0, 1, 3) # (L, B, N, H)
            h_flat = h_perm.reshape(self.agent.rnn_layers, B * N, self.hidden_dim)

            # Prepare Inputs (Sequence Length = 1)
            obs_seq = obs_flat.unsqueeze(1)  # (B*N, 1, C, H, W) [1:Seq_Length]
            agent_ids = torch.arange(N, device=self.device).repeat(B)
            agent_ids_seq = agent_ids.unsqueeze(1)  # (B*N, 1)

            q_values_seq, h_out_flat = self.agent(obs_seq, h_flat, agent_id=agent_ids_seq)
            # q_values_seq: (B*N, 1, n_actions)
            # h_out_flat: (L, B*N, H)

            q_values_flat = q_values_seq.squeeze(1) # (B*N, n_actions)
            q_values = q_values_flat.view(B, N, -1) # (B, N, n_actions)

            # Process Output Hidden: (L, B*N, H) -> (B, N, L, H)
            h_out = h_out_flat.view(self.agent.rnn_layers, B, N, self.hidden_dim)
            next_hidden_state = h_out.permute(1, 2, 0, 3)

            exploit_actions = q_values.argmax(dim=-1).cpu().numpy() # (B, N)
            random_actions = np.random.randint(0, self.n_actions, size=(batch_size, N))
            final_actions = np.where(explore_mask, random_actions, exploit_actions)

            return final_actions, next_hidden_state

        else:
            # Independent Parameters
            actions = []
            next_h_list = []
            
            for i in range(self.n_agents):
                agent_obs = obs_tensor[:, i] # (B, C, H, W)
                
                # Hidden: (B, L, H) -> (L, B, H)
                agent_h = hidden_state[:, i].permute(1, 0, 2).contiguous()

                agent_obs_seq = agent_obs.unsqueeze(1) # (B, 1, ...) [1:Seq_Length]
                
                net = self.agents[i]
                q_values_seq, h_out = net(agent_obs_seq, agent_h) 
                # q_values_seq: (B, 1, n_actions)
                # h_out: (L, B, H)

                q_values = q_values_seq.squeeze(1) # (B, n_actions)
                exploit = q_values.argmax(dim=-1).cpu().numpy()
                random_act = np.random.randint(0, self.n_actions, size=batch_size)
                
                mask_i = explore_mask[:, i]
                chosen = np.where(mask_i, random_act, exploit)
                
                actions.append(chosen)
                # Store hidden as (B, L, H) for stacking
                next_h_list.append(h_out.permute(1, 0, 2))  # (B, L, H)

            final_actions = np.stack(actions, axis=1) # (B, N)
            next_hidden_state = torch.stack(next_h_list, dim=1) # (B, N, L, H)
            return final_actions, next_hidden_state

    def update(self, sample):
        """
        Function: Update network parameters using VDN algorithm.
        Role: Calculate global Q-value (Q_tot) and target Q-value (Target Q_tot), compute TD error and MSE loss, and perform backpropagation.

        Inputs:
        - sample: Dictionary containing training data (from ReplayBuffer).
          Structure:
            - 'global': {'mask': ..., 'dones': ...}
            - 'all_agents': {'obs': ..., 'next_obs': ..., 'actions': ..., 'rewards': ..., 'init_hidden': ..., 'init_target_hidden': ...}

        Outputs:
        - loss: Training loss value (float).
        """
        mask = sample['global']['mask'] # (B, L, 1)
        mask_sum = torch.clamp(mask.sum(), min=1.0)

        init_hidden = sample['all_agents']['init_hidden']
        target_init_hidden = sample['all_agents']['init_target_hidden']

        # Get Global Dones for Target Calculation
        global_dones = sample['global']['dones']

        if self.share_params:
            all_data = sample['all_agents']
            obs = all_data['obs']
            next_obs = all_data['next_obs']
            actions = all_data['actions'].long()
            rewards = all_data['rewards']
            
            B, L, N, C, H, W = obs.shape
            
            # Flatten to (B*N, L, ...)
            # We must preserve sequence order for RNN, but stack agents into batch
            # permute (B, L, N) -> (B, N, L). reshape -> (B*N, L)
            
            obs_flat = obs.permute(0, 2, 1, 3, 4, 5).reshape(B*N, L, C, H, W)
            next_obs_flat = next_obs.permute(0, 2, 1, 3, 4, 5).reshape(B*N, L, C, H, W)
            
            # Hidden State Preparation
            # init_hidden is (L, B, N, H). 
            # We want (L, B*N, H) where B*N follows the order of obs_flat (env0_ag0, env0_ag1...)
            # Since init_hidden is (L, B, N), reshaping last two dims matches (B, N).
            hidden_flat = init_hidden.reshape(self.agent.rnn_layers, B*N, self.hidden_dim)
            target_hidden_flat = target_init_hidden.reshape(self.agent.rnn_layers, B*N, self.hidden_dim)
            
            # Agent IDs
            ids = torch.arange(N, device=self.device).repeat(B) # (B*N)
            ids_seq = ids.unsqueeze(1).expand(-1, L) # (B*N, L)

            # Forward
            q_vals_flat, _ = self.agent(obs_flat, hidden_flat, agent_id=ids_seq)
            with torch.no_grad():
                target_q_vals_flat, _ = self.target_agent(next_obs_flat, target_hidden_flat, agent_id=ids_seq)

            # Reshape back to (B, N, L, n_actions) to sum over agents
            q_vals = q_vals_flat.view(B, N, L, -1)
            target_q_vals = target_q_vals_flat.view(B, N, L, -1)
            
            # Select Actions: actions is (B, L, N) -> need (B, N, L)
            actions_ind = actions.permute(0, 2, 1).unsqueeze(-1)
            
            q_values_selected = q_vals.gather(-1, actions_ind).squeeze(-1) # (B, N, L)
            
            # Target Max
            max_target_q = target_q_vals.max(dim=-1)[0] # (B, N, L)
            
            # VDN Sum
            q_tot = q_values_selected.sum(dim=1).unsqueeze(-1) # (B, L, 1)
            target_q_tot = max_target_q.sum(dim=1).unsqueeze(-1)
            
            total_reward = rewards.sum(dim=2).unsqueeze(-1) # (B, L, 1)

        else:
            # Independent (Logic similar to original but with fixed shapes)
            all_q_values = []
            all_target_q_values = []
            all_rewards = []
            
            for i in range(self.n_agents):
                obs_i = sample['all_agents']['obs'][:, :, i]
                next_obs_i = sample['all_agents']['next_obs'][:, :, i]
                act_i = sample['all_agents']['actions'][:, :, i].long().unsqueeze(-1)
                rew_i = sample['all_agents']['rewards'][:, :, i].unsqueeze(-1)
                
                all_rewards.append(rew_i)

                # Hidden: init_hidden is (L, B, N, H). Slice N -> (L, B, H)
                h_i = init_hidden[:, :, i, :].contiguous()
                target_h_i = target_init_hidden[:, :, i, :].contiguous()

                q_vals, _ = self.agents[i](obs_i, h_i)
                q_sel = q_vals.gather(-1, act_i)
                all_q_values.append(q_sel)
                
                with torch.no_grad():
                    t_q_vals, _ = self.target_agents[i](next_obs_i, target_h_i)
                    max_t = t_q_vals.max(dim=-1, keepdim=True)[0]
                    all_target_q_values.append(max_t)

            q_tot = torch.stack(all_q_values).sum(dim=0) # (B, L, 1)
            target_q_tot = torch.stack(all_target_q_values).sum(dim=0)
            total_reward = torch.stack(all_rewards).sum(dim=0)

        # Compute Loss
        target = total_reward + self.gamma * (1 - global_dones) * target_q_tot
        td_error = (q_tot - target.detach()) ** 2
        masked_loss = td_error * mask
        loss = masked_loss.sum() / mask_sum

        self.optimizer.zero_grad()
        loss.backward()
        params = self.agent.parameters() if self.share_params else self.agents.parameters()
        torch.nn.utils.clip_grad_norm_(params, 10.0) # Standard grad clip
        self.optimizer.step()

        self.soft_update()
        
        return loss.item()
    
    def soft_update(self):
        """
        Function: Soft update target network parameters.
        Role: Synchronize current network parameters to target network with a small ratio (tau) to stabilize training.
        """
        if self.share_params:
            for target_param, local_param in zip(self.target_agent.parameters(), self.agent.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        else:
            for i in range(self.n_agents):
                for target_param, local_param in zip(self.target_agents[i].parameters(), self.agents[i].parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        """
        Function: Save model parameters.
        Input: path (str) - Save path.
        """
        if self.share_params:
            torch.save(self.agent.state_dict(), path)
        else:
            torch.save(self.agents.state_dict(), path)

    def load(self, path):
        """
        Function: Load model parameters.
        Input: path (str) - Load path.
        """
        if self.share_params:
            self.agent.load_state_dict(torch.load(path))
            self.target_agent.load_state_dict(self.agent.state_dict())
        else:
            self.agents.load_state_dict(torch.load(path))
            self.target_agents.load_state_dict(self.agents.state_dict())

    def train(self):
        """Function: Set networks to training mode."""
        if self.share_params:
            self.agent.train()
        else:
            self.agents.train()

    def eval(self):
        """Function: Set networks to evaluation mode."""
        if self.share_params:
            self.agent.eval()
        else:
            self.agents.eval()