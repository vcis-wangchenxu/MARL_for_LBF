from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from networks.DRQN import AgentDRQN
from networks.QMIX_Net import QMixer

class QMIX:
    """
    QMIX Algorithm.
    Core idea: Q_tot(s, a) = MixingNet(Q_1, ..., Q_n, s)
    Constraint: dQ_tot / dQ_i >= 0 (Monotonicity)
    """
    def __init__(self, env_info: dict, args: Dict):
        self.env_info = env_info
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        self.obs_shape = env_info['obs_shape']
        self.state_shape = env_info['state_shape']

        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq

        self.hidden_dim = args.rnn_hidden_dim
        self.norm_factor = args.norm_factor
        self.tau = args.target_tau
        self.share_params = args.share_parameters

        # Epsilon-Greedy
        self.epsilon_start = args.epsilon_start
        self.epsilon_end = args.epsilon_end
        self.epsilon_decay = args.epsilon_decay
        self.epsilon = self.epsilon_start
        
        self.device = torch.device(args.device if hasattr(args, 'device') else 'cpu')

        # --- Agent Networks ---
        if self.share_params:
            print(f"[QMIX] Parameter Sharing Enabled. All {self.n_agents} agents share one DRQN.")
            self.agent = AgentDRQN(self.obs_shape, self.n_actions, 
                                   self.n_agents, self.hidden_dim,  
                                   norm_factor=self.norm_factor, 
                                   use_agent_id=True).to(self.device)

            self.target_agent = AgentDRQN(self.obs_shape, self.n_actions, 
                                          self.n_agents, self.hidden_dim, 
                                          norm_factor=self.norm_factor, 
                                          use_agent_id=True).to(self.device)

            self.target_agent.load_state_dict(self.agent.state_dict())
        else:
            print(f"[QMIX] Parameter Sharing Disabled. Creating {self.n_agents} independent DRQNs.")
            self.agents = nn.ModuleList([
                AgentDRQN(self.obs_shape, self.n_actions, 
                          1, self.hidden_dim, 
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

        # --- Mixing Network ---
        self.mixer = QMixer(self.n_agents, self.state_shape, 
                            mixing_embed_dim=getattr(args, 'mixing_embed_dim', 32),
                            hypernet_embed=getattr(args, 'hypernet_embed', 64)).to(self.device)

        self.target_mixer = QMixer(self.n_agents, self.state_shape, 
                                   mixing_embed_dim=getattr(args, 'mixing_embed_dim', 32),
                                   hypernet_embed=getattr(args, 'hypernet_embed', 64)).to(self.device)
        
        self.target_mixer.load_state_dict(self.mixer.state_dict())

        # Optimizer
        self.params = list(self.mixer.parameters())
        if self.share_params:
            self.params += list(self.agent.parameters())
        else:
            self.params += list(self.agents.parameters())
            
        self.optimizer = optim.Adam(self.params, lr=self.lr)
        self.criterion = nn.MSELoss()

    def init_hidden(self, batch_size=1) -> torch.Tensor:
        if self.share_params:
            h = self.agent.init_hidden(batch_size * self.n_agents, device=self.device)
            h = h.view(self.agent.rnn_layers, batch_size, self.n_agents, self.hidden_dim)
            h = h.permute(1, 2, 0, 3).contiguous()
            return h
        else:
            h_list = [agent.init_hidden(batch_size, device=self.device) for agent in self.agents]
            h_stack = torch.stack(h_list, dim=0) 
            h = h_stack.permute(2, 0, 1, 3).contiguous() 
            return h

    @torch.no_grad()
    def take_action(self, obs_tensor, hidden_state, current_step, evaluation=False):
        batch_size = obs_tensor.shape[0]

        if evaluation:
            self.epsilon = 0.0
            explore_mask = np.zeros((batch_size, self.n_agents), dtype=bool)
        else:
            self.epsilon = max(self.epsilon_end, self.epsilon_start - \
                           (self.epsilon_start - self.epsilon_end) * (current_step / self.epsilon_decay))
            rand_probs = np.random.rand(batch_size, self.n_agents)
            explore_mask = rand_probs < self.epsilon

        if self.share_params:
            B, N, C, H, W = obs_tensor.shape
            obs_flat = obs_tensor.view(B * N, C, H, W)
            
            h_perm = hidden_state.permute(2, 0, 1, 3) 
            h_flat = h_perm.reshape(self.agent.rnn_layers, B * N, self.hidden_dim)

            obs_seq = obs_flat.unsqueeze(1)
            agent_ids = torch.arange(N, device=self.device).repeat(B)
            agent_ids_seq = agent_ids.unsqueeze(1)

            q_values_seq, h_out_flat = self.agent(obs_seq, h_flat, agent_id=agent_ids_seq)
            
            q_values_flat = q_values_seq.squeeze(1)
            q_values = q_values_flat.view(B, N, -1)

            h_out = h_out_flat.view(self.agent.rnn_layers, B, N, self.hidden_dim)
            next_hidden_state = h_out.permute(1, 2, 0, 3)

            exploit_actions = q_values.argmax(dim=-1).cpu().numpy()
            random_actions = np.random.randint(0, self.n_actions, size=(batch_size, N))
            final_actions = np.where(explore_mask, random_actions, exploit_actions)

            return final_actions, next_hidden_state

        else:
            actions = []
            next_h_list = []
            
            for i in range(self.n_agents):
                agent_obs = obs_tensor[:, i]
                agent_h = hidden_state[:, i].permute(1, 0, 2).contiguous()
                agent_obs_seq = agent_obs.unsqueeze(1)
                
                net = self.agents[i]
                q_values_seq, h_out = net(agent_obs_seq, agent_h) 
                
                q_values = q_values_seq.squeeze(1)
                exploit = q_values.argmax(dim=-1).cpu().numpy()
                random_act = np.random.randint(0, self.n_actions, size=batch_size)
                
                mask_i = explore_mask[:, i]
                chosen = np.where(mask_i, random_act, exploit)
                
                actions.append(chosen)
                next_h_list.append(h_out.permute(1, 0, 2)) 

            final_actions = np.stack(actions, axis=1)
            next_hidden_state = torch.stack(next_h_list, dim=1)
            return final_actions, next_hidden_state

    def update(self, sample):
        mask = sample['global']['mask'] # (B, L, 1)
        mask_sum = torch.clamp(mask.sum(), min=1.0)

        init_hidden = sample['all_agents']['init_hidden']
        target_init_hidden = sample['all_agents']['init_target_hidden']
        global_dones = sample['global']['dones']
        
        # QMIX needs State
        state = sample['global']['state'] # (B, L, *State_Shape)
        next_state = sample['global']['next_state']

        if self.share_params:
            all_data = sample['all_agents']
            obs = all_data['obs']
            next_obs = all_data['next_obs']
            actions = all_data['actions'].long()
            rewards = all_data['rewards']
            
            B, L, N, C, H, W = obs.shape
            
            obs_flat = obs.permute(0, 2, 1, 3, 4, 5).reshape(B*N, L, C, H, W)
            next_obs_flat = next_obs.permute(0, 2, 1, 3, 4, 5).reshape(B*N, L, C, H, W)
            
            hidden_flat = init_hidden.reshape(self.agent.rnn_layers, B*N, self.hidden_dim)
            target_hidden_flat = target_init_hidden.reshape(self.agent.rnn_layers, B*N, self.hidden_dim)
            
            ids = torch.arange(N, device=self.device).repeat(B)
            ids_seq = ids.unsqueeze(1).expand(-1, L)

            q_vals_flat, _ = self.agent(obs_flat, hidden_flat, agent_id=ids_seq)
            with torch.no_grad():
                target_q_vals_flat, _ = self.target_agent(next_obs_flat, target_hidden_flat, agent_id=ids_seq)

            q_vals = q_vals_flat.view(B, N, L, -1).permute(0, 2, 1, 3) # (B, L, N, n_actions)
            target_q_vals = target_q_vals_flat.view(B, N, L, -1).permute(0, 2, 1, 3)
            
            actions_ind = actions.unsqueeze(-1) # (B, L, N, 1)
            
            q_values_selected = q_vals.gather(-1, actions_ind).squeeze(-1) # (B, L, N)
            max_target_q = target_q_vals.max(dim=-1)[0] # (B, L, N)
            
            total_reward = rewards.sum(dim=2).unsqueeze(-1) # (B, L, 1)

        else:
            all_q_values = []
            all_target_q_values = []
            all_rewards = []
            
            for i in range(self.n_agents):
                obs_i = sample['all_agents']['obs'][:, :, i]
                next_obs_i = sample['all_agents']['next_obs'][:, :, i]
                act_i = sample['all_agents']['actions'][:, :, i].long().unsqueeze(-1)
                rew_i = sample['all_agents']['rewards'][:, :, i].unsqueeze(-1)
                
                all_rewards.append(rew_i)

                h_i = init_hidden[:, :, i, :].contiguous()
                target_h_i = target_init_hidden[:, :, i, :].contiguous()

                q_vals, _ = self.agents[i](obs_i, h_i)
                q_sel = q_vals.gather(-1, act_i)
                all_q_values.append(q_sel)
                
                with torch.no_grad():
                    t_q_vals, _ = self.target_agents[i](next_obs_i, target_h_i)
                    max_t = t_q_vals.max(dim=-1, keepdim=True)[0]
                    all_target_q_values.append(max_t)

            q_values_selected = torch.cat(all_q_values, dim=2) # (B, L, N)
            max_target_q = torch.cat(all_target_q_values, dim=2)
            total_reward = torch.stack(all_rewards).sum(dim=0)

        # --- Mixing ---
        # q_values_selected: (B, L, N)
        # state: (B, L, *State_Shape)
        q_tot = self.mixer(q_values_selected, state) # (B, L, 1)
        
        with torch.no_grad():
            target_q_tot = self.target_mixer(max_target_q, next_state) # (B, L, 1)

        # Compute Loss
        target = total_reward + self.gamma * (1 - global_dones) * target_q_tot
        td_error = (q_tot - target.detach()) ** 2
        masked_loss = td_error * mask
        loss = masked_loss.sum() / mask_sum

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()

        self.soft_update()
        
        return loss.item()
    
    def soft_update(self):
        # Update Mixer
        for target_param, local_param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
            
        # Update Agents
        if self.share_params:
            for target_param, local_param in zip(self.target_agent.parameters(), self.agent.parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
        else:
            for i in range(self.n_agents):
                for target_param, local_param in zip(self.target_agents[i].parameters(), self.agents[i].parameters()):
                    target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        state_dict = {
            'mixer': self.mixer.state_dict(),
            'target_mixer': self.target_mixer.state_dict()
        }
        if self.share_params:
            state_dict['agent'] = self.agent.state_dict()
        else:
            state_dict['agents'] = self.agents.state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        
        if self.share_params:
            self.agent.load_state_dict(checkpoint['agent'])
            self.target_agent.load_state_dict(self.agent.state_dict())
        else:
            self.agents.load_state_dict(checkpoint['agents'])
            self.target_agents.load_state_dict(self.agents.state_dict())

    def train(self):
        self.mixer.train()
        if self.share_params:
            self.agent.train()
        else:
            self.agents.train()

    def eval(self):
        self.mixer.eval()
        if self.share_params:
            self.agent.eval()
        else:
            self.agents.eval()
