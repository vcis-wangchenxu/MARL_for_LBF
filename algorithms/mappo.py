import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

from networks.DRQN import AgentDRQN

class MAPPO:
    """
    MAPPO Algorithm (Centralized Training Decentralized Execution).
    Actor takes local observations (obs).
    Critic takes global state (state).
    Both use DRQN (CNN+GRU).
    """
    def __init__(self, env_info, args):
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        self.obs_shape = env_info['obs_shape']
        self.state_shape = env_info['state_shape'] # (N*C, H, W)
        
        self.args = args
        self.device = torch.device(args.device)
        self.share_params = args.share_parameters
        
        # Hyperparameters
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

        # Split Hidden Dim for Actor and Critic
        # Buffer expects 'rnn_hidden_dim', so we use half for actor, half for critic
        total_hidden_dim = args.rnn_hidden_dim
        self.actor_hidden_dim = total_hidden_dim // 2
        self.critic_hidden_dim = total_hidden_dim - self.actor_hidden_dim
        self.rnn_layers = args.rnn_layers

        print(f"[MAPPO] Init. Actor Hidden: {self.actor_hidden_dim}, Critic Hidden: {self.critic_hidden_dim}")

        # --- Actor (Policy) ---
        # Input: Obs (C, H, W) -> Output: n_actions
        self.actor = AgentDRQN(
            self.obs_shape, 
            self.n_actions, 
            self.n_agents, 
            self.actor_hidden_dim,
            rnn_layers=self.rnn_layers,
            use_agent_id=True, # Actor needs ID
            norm_factor=args.norm_factor
        ).to(self.device)

        # --- Critic (Value) ---
        # Input: State (N*C, H, W) -> Output: 1 Value
        self.critic = AgentDRQN(
            self.state_shape, 
            1, # Value dimension
            self.n_agents, 
            self.critic_hidden_dim,
            rnn_layers=self.rnn_layers,
            use_agent_id=True, # Critic needs ID to differentiate agents' value contribution
            norm_factor=args.norm_factor
        ).to(self.device)

        # Joint Optimizer
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=self.lr
        )

    def init_hidden(self, batch_size):
        """
        Returns stacked hidden states for both Actor and Critic.
        Shape: (Batch, N, Layers, Total_Hidden)
        """
        # Actor Hidden: (L, B*N, H_a)
        h_a = self.actor.init_hidden(batch_size * self.n_agents, self.device)
        # Critic Hidden: (L, B*N, H_c)
        h_c = self.critic.init_hidden(batch_size * self.n_agents, self.device)
        
        # Reshape to (L, B, N, H)
        h_a = h_a.view(self.rnn_layers, batch_size, self.n_agents, self.actor_hidden_dim)
        h_c = h_c.view(self.rnn_layers, batch_size, self.n_agents, self.critic_hidden_dim)
        
        # Concatenate along hidden dim -> (L, B, N, Total_H)
        h_total = torch.cat([h_a, h_c], dim=-1)
        
        # Permute to (B, N, L, Total_H) for buffer storage
        return h_total.permute(1, 2, 0, 3).contiguous()

    def take_action(self, obs, hidden_state, evaluation=False):
        """
        MAPPO Rollout.
        Critic needs State. Since train_onpolicy.py only passes Obs, we reconstruct State from Obs.
        """
        B, N, C, H, W = obs.shape
        obs = obs.float()

        # Reconstruct Global State from Obs (LBF Specific: Flatten N agents' obs)
        # Obs: (B, N, C, H, W) -> State: (B, N*C, H, W)
        state = obs.view(B, -1, H, W)
        
        # Split Hidden State
        # hidden_state: (B, N, L, Total_H)
        h_in = hidden_state.permute(2, 0, 1, 3) # (L, B, N, Total_H)
        h_a_in = h_in[..., :self.actor_hidden_dim].contiguous()
        h_c_in = h_in[..., self.actor_hidden_dim:].contiguous()
        
        # Reshape for Network: (L, B*N, H)
        h_a_flat = h_a_in.view(self.rnn_layers, B * N, self.actor_hidden_dim)
        h_c_flat = h_c_in.view(self.rnn_layers, B * N, self.critic_hidden_dim)

        # Prepare Inputs
        # Actor Inputs
        obs_flat = obs.view(B * N, C, H, W).unsqueeze(1) # (B*N, 1, C, H, W)
        agent_ids = torch.arange(N, device=self.device).repeat(B)
        agent_ids_seq = agent_ids.unsqueeze(1) # (B*N, 1)

        # Critic Inputs
        # State needs to be repeated for each agent: (B, N*C, H, W) -> (B, N, N*C, H, W) -> (B*N, 1, N*C, H, W)
        state_repeated = state.unsqueeze(1).repeat(1, N, 1, 1, 1)
        state_flat = state_repeated.view(B * N, -1, H, W).unsqueeze(1)

        # Forward
        # Actor -> Logits
        logits_seq, h_a_next = self.actor(obs_flat, h_a_flat, agent_id=agent_ids_seq)
        logits = logits_seq.squeeze(1) # (B*N, n_actions)
        
        # Critic -> Values
        values_seq, h_c_next = self.critic(state_flat, h_c_flat, agent_id=agent_ids_seq)
        values = values_seq.squeeze(1).squeeze(-1) # (B*N)

        # Action Selection
        if evaluation:
            actions_flat = logits.argmax(dim=-1)
            log_probs_flat = torch.zeros_like(actions_flat).float()
        else:
            dist = Categorical(logits=logits)
            actions_flat = dist.sample()
            log_probs_flat = dist.log_prob(actions_flat)

        # Restore Shapes
        actions = actions_flat.view(B, N)
        log_probs = log_probs_flat.view(B, N)
        values = values.view(B, N)
        
        # Merge Hidden States
        h_a_next = h_a_next.view(self.rnn_layers, B, N, self.actor_hidden_dim)
        h_c_next = h_c_next.view(self.rnn_layers, B, N, self.critic_hidden_dim)
        next_hidden = torch.cat([h_a_next, h_c_next], dim=-1).permute(1, 2, 0, 3).contiguous()

        return actions, log_probs, values, next_hidden

    def update(self, buffer):
        data = buffer.get_data()
        rewards = data['rewards']
        values = data['values']
        masks = data['masks']
        
        # GAE Calculation
        advs = np.zeros_like(rewards)
        last_gae_lam = 0
        T = buffer.buffer_size
        
        for t in reversed(range(T)):
            delta = rewards[t] + self.args.gamma * values[t+1] * masks[t] - values[t]
            advs[t] = last_gae_lam = delta + self.args.gamma * self.args.gae_lambda * masks[t] * last_gae_lam
            
        advantages = advs
        
        stats = {'loss_value': 0, 'loss_action': 0, 'loss_entropy': 0}
        update_count = 0
        
        for _ in range(self.ppo_epochs):
            data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            
            for sample in data_generator:
                obs_batch, hidden_batch, actions_batch, values_batch, \
                return_batch, old_action_log_probs_batch, adv_targ, \
                masks_batch, agent_ids_batch, state_batch = sample
                
                # Input Shapes: (B, L, N, ...)
                B, L, N, C, H, W = obs_batch.shape
                
                obs_in = obs_batch.transpose(1, 2).reshape(B*N, L, C, H, W) # (B*N, L, C, H, W)
                ids_in = agent_ids_batch.transpose(1, 2).reshape(B*N, L)
                
                # Prepare State for Critic: (B, L, State_Dim) -> Expand to (B, L, N, State_Dim)
                # state_batch: (B, L, State_Dim_C, H, W)
                # We need to broadcast state to all agents in the batch
                state_in = state_batch.unsqueeze(2).repeat(1, 1, N, 1, 1, 1) # (B, L, N, ...)
                state_in = state_in.transpose(1, 2).reshape(B*N, L, -1, H, W)
                
                # Split Hidden
                # hidden_batch: (L, B, N, Total_H) -> (L, B*N, Total_H)
                h_in = hidden_batch.reshape(self.rnn_layers, B*N, -1)
                h_a_in = h_in[..., :self.actor_hidden_dim].contiguous()
                h_c_in = h_in[..., self.actor_hidden_dim:].contiguous()

                # Actor
                logits, _ = self.actor(obs_in, h_a_in, agent_id=ids_in)
                logits_flat = logits.reshape(-1, self.n_actions)
                
                # Critic
                values_pred, _ = self.critic(state_in, h_c_in, agent_id=ids_in)
                values_pred_flat = values_pred.reshape(-1)

                # Loss Calculation
                actions_flat = actions_batch.transpose(1, 2).reshape(-1)
                old_log_probs_flat = old_action_log_probs_batch.transpose(1, 2).reshape(-1)
                returns_flat = return_batch.transpose(1, 2).reshape(-1)
                adv_targ_flat = adv_targ.transpose(1, 2).reshape(-1)
                
                # Policy Loss
                dist = Categorical(logits=logits_flat)
                action_log_probs = dist.log_prob(actions_flat)
                dist_entropy = dist.entropy().mean()
                
                ratio = torch.exp(action_log_probs - old_log_probs_flat)
                surr1 = ratio * adv_targ_flat
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_flat
                action_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = F.mse_loss(returns_flat, values_pred_flat)
                
                loss = action_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], self.max_grad_norm)
                self.optimizer.step()
                
                stats['loss_value'] += value_loss.item()
                stats['loss_action'] += action_loss.item()
                stats['loss_entropy'] += dist_entropy.item()
                update_count += 1

        return {k: v / update_count for k, v in stats.items()}

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
            
    def train(self):
        self.actor.train()
        self.critic.train()
            
    def eval(self):
        self.actor.eval()
        self.critic.eval()