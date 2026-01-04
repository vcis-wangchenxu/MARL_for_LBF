import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from networks.ActorCritic import ActorCritic

class IPPO:
    """
    Independent PPO (IPPO) Algorithm.
    Function: Implements Independent PPO algorithm for multi-agent environments. Each agent can learn independently or share parameters.
    Role: Manages policy networks (Actor-Critic), performs action selection, and updates network parameters based on the PPO algorithm.
    """
    def __init__(self, env_info, args):
        """
        Initialize IPPO Algorithm.

        Parameters:
        - env_info: Environment information dictionary, containing:
            - n_agents: Number of agents
            - n_actions: Size of action space
            - obs_shape: Shape of observation space
        - args: Arguments object containing hyperparameter configurations (e.g., lr, gamma, clip_param, hidden_dim, etc.).
        """
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        self.obs_shape = env_info['obs_shape']
        
        self.args = args
        self.device = torch.device(args.device)
        self.share_params = args.share_parameters
        
        # Hyperparameters
        self.hidden_dim = args.rnn_hidden_dim
        self.rnn_layers = args.rnn_layers
        self.lr = args.lr
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self.clip_param = args.clip_param
        self.ppo_epochs = args.ppo_epochs
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.data_chunk_length = args.data_chunk_length

        # Init Networks
        if self.share_params:
            print(f"[IPPO] Parameter Sharing Enabled. All {self.n_agents} agents share one ActorCritic ({self.rnn_layers} layers).")
            self.agent = ActorCritic(
                self.obs_shape, 
                self.n_actions, 
                self.n_agents, 
                self.hidden_dim,
                rnn_layers=self.rnn_layers,
                use_agent_id=True
            ).to(self.device)
            self.optimizer = optim.Adam(self.agent.parameters(), lr=self.lr)
            
        else:
            print(f"[IPPO] Parameter Sharing Disabled. Creating {self.n_agents} independent ActorCritics ({self.rnn_layers} layers).")
            self.agents = nn.ModuleList([
                ActorCritic(
                    self.obs_shape, 
                    self.n_actions, 
                    self.n_agents, 
                    self.hidden_dim,
                    rnn_layers=self.rnn_layers,
                    use_agent_id=False
                ) for _ in range(self.n_agents)
            ]).to(self.device)
            self.optimizer = optim.Adam(self.agents.parameters(), lr=self.lr)

    def init_hidden(self, batch_size):
        """
        Function: Initialize RNN hidden states.
        Role: Generate zero-filled hidden states for a new episode or batch.

        Inputs:
        - batch_size: Batch size (int).

        Outputs:
        - hidden_state: Initialized hidden state.
          Type: torch.Tensor
          Shape: (Batch, N_Agents, Layers, Hidden)
        """
        if self.share_params:
            # agent.init_hidden returns (Layers, Batch*N, Hidden)
            h = self.agent.init_hidden(batch_size * self.n_agents, self.device)
            # Reshape to (Layers, Batch, N, Hidden)
            h = h.view(self.rnn_layers, batch_size, self.n_agents, self.hidden_dim)
            # Permute to (Batch, N, Layers, Hidden)
            return h.permute(1, 2, 0, 3).contiguous()
        else:
            # List of (Layers, Batch, Hidden)
            h_list = [agent.init_hidden(batch_size, self.device) for agent in self.agents]
            # Stack to (Layers, Batch, N, Hidden) -> Permute to (Batch, N, Layers, Hidden)
            h_stack = torch.stack(h_list, dim=2) 
            return h_stack.permute(1, 2, 0, 3).contiguous()

    def take_action(self, obs, hidden_state, evaluation=False):
        """
        Function: Select actions based on observations and hidden states.
        Role: Forward pass through Actor-Critic network, outputting actions, log probabilities, value estimates, and new hidden states.

        Inputs:
        - obs: Observation data.
          Type: torch.Tensor
          Shape: (Batch, N_Agents, *obs_shape) (e.g., (B, N, C, H, W)) [B: num_envs]
        - hidden_state: RNN hidden state.
          Type: torch.Tensor
          Shape: (Batch, N_Agents, Layers, Hidden)
        - evaluation: Whether in evaluation mode (bool). If True, select action with highest probability; otherwise sample.

        Outputs:
        - actions: Selected actions. (Batch, N_Agents)
        - log_probs: Log probabilities of actions. (Batch, N_Agents)
        - values: State value estimates. (Batch, N_Agents)
        - next_hidden: Updated hidden states. (Batch, N_Agents, Layers, Hidden)
        """
        B, N, C, H, W = obs.shape
        obs = obs.float()
        
        if self.share_params:
            obs_flat = obs.view(B * N, C, H, W)
            obs_seq = obs_flat.unsqueeze(1) # (B*N, 1, ...) [1:Seq_Length]

            agent_ids = torch.arange(N, device=self.device).repeat(B)
            agent_ids_seq = agent_ids.unsqueeze(1) # (B, N, 1)

            # Hidden: (B, N, L, H) -> (L, B, N, H) -> (L, B*N, H)
            h_in = hidden_state.permute(2, 0, 1, 3).reshape(self.rnn_layers, B * N, self.hidden_dim)

            logits_seq, values_seq, next_hidden_flat = self.agent(obs_seq, h_in, agent_id=agent_ids_seq)
            # logits_seq: (B*N, 1, n_actions)
            # values_seq: (B*N, 1, 1)
            # next_hidden_flat: (RNN_Layers, B*N, hidden_dim)

            logits = logits_seq.squeeze(1)  # (B*N, n_actions)
            values_flat = values_seq.squeeze(1) # (B*N)

            if evaluation:
                actions_flat = logits.argmax(dim=-1) # (B*N,)
                log_probs_flat = torch.zeros_like(actions_flat).float()
            else:
                dist = Categorical(logits=logits)
                actions_flat = dist.sample()
                log_probs_flat = dist.log_prob(actions_flat)

            actions = actions_flat.view(B, N)
            log_probs = log_probs_flat.view(B, N)
            values = values_flat.view(B, N)
            
            # Restore Hidden: (L, B*N, H) -> (L, B, N, H) -> (B, N, L, H)
            next_hidden = next_hidden_flat.view(self.rnn_layers, B, N, self.hidden_dim).permute(1, 2, 0, 3).contiguous()
            
            return actions, log_probs, values, next_hidden

        else:
            actions_list = []
            log_probs_list = []
            values_list = []
            next_hidden_list = []

            for i in range(self.n_agents):
                agent_obs = obs[:, i] 
                # Hidden: (B, N, L, H) -> (B, L, H) -> (L, B, H)
                agent_h = hidden_state[:, i, :, :].permute(1, 0, 2).contiguous()

                agent_obs_seq = agent_obs.unsqueeze(1)
                
                logits_seq, val_seq, next_h = self.agents[i](agent_obs_seq, agent_h)
                
                logits = logits_seq.squeeze(1)
                val = val_seq.squeeze(1)

                if evaluation:
                    act = logits.argmax(dim=-1)
                    log_p = torch.zeros_like(act).float()
                else:
                    dist = Categorical(logits=logits)
                    act = dist.sample()
                    log_p = dist.log_prob(act)

                actions_list.append(act)
                log_probs_list.append(log_p)
                values_list.append(val.squeeze(-1))
                # next_h: (L, B, H) -> (B, L, H)
                next_hidden_list.append(next_h.permute(1, 0, 2))

            actions = torch.stack(actions_list, dim=1)
            log_probs = torch.stack(log_probs_list, dim=1)
            values = torch.stack(values_list, dim=1)
            
            # Stack: (B, N, L, H)
            next_hidden = torch.stack(next_hidden_list, dim=1)

            return actions, log_probs, values, next_hidden

    def update(self, buffer):
        """
        Function: Update network parameters using the PPO algorithm.
        Role: Calculate GAE advantages, generate mini-batches, compute PPO losses (Action, Value, Entropy), and perform backpropagation.

        Inputs:
        - buffer: Buffer object storing trajectory data (RolloutBuffer).

        Outputs:
        - train_info: Dictionary containing training metrics.
            - "loss_value": Average value loss (float)
            - "loss_action": Average policy loss (float)
            - "loss_entropy": Average entropy loss (float)
        """
        data = buffer.get_data()
        rewards = data['rewards'] 
        values = data['values']   
        masks = data['masks']     
        
        advs = np.zeros_like(rewards)
        last_gae_lam = 0
        T = buffer.buffer_size
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.args.gamma * values[t+1] * masks[t] - values[t]
            advs[t] = last_gae_lam = delta + self.args.gamma * self.args.gae_lambda * masks[t] * last_gae_lam
            
        advantages = advs
        
        total_loss_value = 0
        total_loss_action = 0
        total_loss_entropy = 0
        update_count = 0
        
        for _ in range(self.ppo_epochs):
            # Generator yields hidden_batch as (Layers, Batch, N, H) due to RolloutBuffer internals
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            
            for sample in data_generator:
                obs_batch, hidden_batch, actions_batch, values_batch, \
                return_batch, old_action_log_probs_batch, adv_targ, masks_batch, agent_ids_batch = sample
                
                B, L, N, C, H, W = obs_batch.shape
                
                if self.share_params:
                    # Reshape Obs: (B, L, N, ...) -> (B, N, L, ...) -> (B*N, L, ...)
                    obs_in = obs_batch.transpose(1, 2).reshape(B*N, L, C, H, W)
                    ids_in = agent_ids_batch.transpose(1, 2).reshape(B*N, L)
                    
                    # (Layers, B, N, H) -> (Layers, B*N, H)
                    h_in = hidden_batch.reshape(self.rnn_layers, B*N, self.hidden_dim)

                    logits, values_pred, _ = self.agent(obs_in, h_in, agent_id=ids_in)
                    
                    actions_flat = actions_batch.transpose(1, 2).reshape(-1)
                    old_log_probs_flat = old_action_log_probs_batch.transpose(1, 2).reshape(-1)
                    returns_flat = return_batch.transpose(1, 2).reshape(-1)
                    adv_targ_flat = adv_targ.transpose(1, 2).reshape(-1)
                    
                    logits_flat = logits.reshape(-1, self.n_actions)
                    values_pred_flat = values_pred.reshape(-1)
                    
                else:
                    logits_list = []
                    values_pred_list = []
                    
                    for i in range(self.n_agents):
                        ag_obs = obs_batch[:, :, i] 
                        # (Layers, B, N, H) -> (B, Layers, H) -> (Layers, B, H)
                        ag_h = hidden_batch[:, :, i, :].contiguous()
                        
                        ag_logits, ag_val, _ = self.agents[i](ag_obs, ag_h)
                        logits_list.append(ag_logits)
                        values_pred_list.append(ag_val)
                    
                    logits_stacked = torch.stack(logits_list, dim=2) 
                    values_stacked = torch.stack(values_pred_list, dim=2)
                    
                    logits_flat = logits_stacked.reshape(-1, self.n_actions)
                    values_pred_flat = values_stacked.reshape(-1)
                    
                    actions_flat = actions_batch.reshape(-1)
                    old_log_probs_flat = old_action_log_probs_batch.reshape(-1)
                    returns_flat = return_batch.reshape(-1)
                    adv_targ_flat = adv_targ.reshape(-1)

                dist = Categorical(logits=logits_flat)
                action_log_probs = dist.log_prob(actions_flat)
                dist_entropy = dist.entropy().mean()
                
                ratio = torch.exp(action_log_probs - old_log_probs_flat)
                surr1 = ratio * adv_targ_flat
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_flat
                action_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(returns_flat, values_pred_flat)
                
                loss = action_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                
                params = self.agent.parameters() if self.share_params else self.agents.parameters()
                nn.utils.clip_grad_norm_(params, self.max_grad_norm)
                self.optimizer.step()
                
                total_loss_value += value_loss.item()
                total_loss_action += action_loss.item()
                total_loss_entropy += dist_entropy.item()
                update_count += 1

        return {
            "loss_value": total_loss_value / update_count,
            "loss_action": total_loss_action / update_count,
            "loss_entropy": total_loss_entropy / update_count
        }

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
        else:
            self.agents.load_state_dict(torch.load(path))
            
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