from typing import List, Tuple, Dict
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

from networks.DRQN import AgentDRQN

class IDQN:
    """
    Independent Deep Recurrent Q-Network (IDQN) 算法。
    每个智能体拥有独立的 Q 网络（CNN + GRU），独立进行学习。
    """
    def __init__(self, env_info: dict, args: Dict):
        self.env_info = env_info
        self.n_agents = env_info['n_agents']
        self.n_actions = env_info['n_actions']
        self.obs_shape = env_info['obs_shape']

        self.args = args
        self.lr = args.lr
        self.gamma = args.gamma
        self.target_update_freq = args.target_update_freq
        
        # 获取超参数 (支持从 cfg 读取，也提供默认值)
        self.hidden_dim = getattr(args, 'rnn_hidden_dim', 64)
        self.norm_factor = getattr(args, 'norm_factor', 10.0)
        self.tau = getattr(args, 'target_tau', 0.01)  # Target 网络软更新系数

        # === Epsilon-Greedy 探索参数 ===
        self.epsilon_start = getattr(args, 'epsilon_start', 1.0)
        self.epsilon_end = getattr(args, 'epsilon_end', 0.05)
        self.epsilon_decay = getattr(args, 'epsilon_decay', 50000)
        self.epsilon = self.epsilon_start  # 当前探索率

        self.device = torch.device(args.device if hasattr(args, 'device') else 'cpu')

        # 初始化 Q 网络和 Target 网络
        self.agents = nn.ModuleList([
            AgentDRQN(self.obs_shape, self.n_actions, self.hidden_dim, self.norm_factor) 
            for _ in range(self.n_agents)
        ]).to(self.device)

        self.target_agents = nn.ModuleList([
            AgentDRQN(self.obs_shape, self.n_actions, self.hidden_dim, self.norm_factor) 
            for _ in range(self.n_agents)
        ]).to(self.device)

        self.target_agents.load_state_dict(self.agents.state_dict())

        self.optimizer = optim.Adam(self.agents.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss(reduction='none')

        self._train_step_count = 0

    def init_hidden(self, batch_size=1) -> torch.Tensor:
        """ 初始化所有智能体的 RNN 隐藏状态 """
        h_list = [agent.init_hidden(batch_size, device=self.device) for agent in self.agents]
        return torch.stack(h_list, dim=1)    # (B, N, Hidden)

    @torch.no_grad()
    def take_action(self, obs_tensor, hidden_state, current_step, evaluation=False) -> Tuple[np.ndarray, torch.Tensor]:
        """
        执行推理动作 (Inference)。
        
        参数:
        - obs_tensor: (1, N, C, H, W) -> Batch=1
        - hidden_state: (1, N, Hidden) -> Batch=1
        - current_step: 当前总训练步数 (用于计算 epsilon 衰减)
        
        返回:
        - actions: (N,) numpy array
        - next_hidden_state: (1, N, Hidden) tensor
        """
        if evaluation:
            self.epsilon = 0.0
        else:
            self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                           np.exp(-1. * current_step / self.epsilon_decay)
        
        actions = []
        next_h_list = []

        for i in range(self.n_agents):
            agent_obs = obs_tensor[:, i]    # (1, C, H, W)
            agent_h = hidden_state[:, i].contiguous()    # (1, Hidden)

            # Epsilon-Greedy
            if random.random() < self.epsilon:
                actions.append(random.randint(0, self.n_actions - 1))
                _, h_out = self.agents[i](agent_obs, agent_h)
            else:
                q_values, h_out = self.agents[i](agent_obs, agent_h)
                action = q_values.argmax(dim=-1).item()
                actions.append(action)   # (N, )
            
            next_h_list.append(h_out)    # (1, Hidden)
        
        next_hidden_state = torch.stack(next_h_list, dim=1)    # (1, N, Hidden)

        return np.array(actions), next_hidden_state
    
    def update(self, sample):
        """
        模型训练 (Training)。
        
        参数:
        - sample: 从 ReplayBuffer.sample(...) 返回的嵌套字典，包含每个智能体的数据段
        """
        mask_np = sample['global']['mask']  # (B, L)
        mask = torch.tensor(mask_np, dtype=torch.float32, device=self.device)
        mask_sum = torch.clamp(mask.sum(), min=1.0)    # 防止除以零
        
        # 初始化hidden_state
        batch_size = mask.shape[0]
        hidden_state = self.init_hidden(batch_size)  # (B, N, Hidden)
        target_hidden_state = self.init_hidden(batch_size)  # (B, N, Hidden)

        total_loss = 0.0

        # 对每个智能体分别计算损失并更新
        for i in range(self.n_agents):
            agent_data = sample[i]
            curr_obs = torch.tensor(agent_data['obs'], dtype=torch.float32, device=self.device)          # (B, L, C, H, W)
            curr_next_obs = torch.tensor(agent_data['next_obs'], dtype=torch.float32, device=self.device)  # (B, L, C, H, W)
            curr_act = torch.tensor(agent_data['actions'], dtype=torch.long, device=self.device).unsqueeze(-1)      # (B, L, 1)
            curr_rew = torch.tensor(agent_data['rewards'], dtype=torch.float32, device=self.device).unsqueeze(-1)    # (B, L, 1)
            curr_done = torch.tensor(agent_data['dones'], dtype=torch.float32, device=self.device).unsqueeze(-1)        # (B, L, 1)

            # 获取当前智能体的隐藏状态
            curr_hidden = hidden_state[:, i].contiguous()      # (B, Hidden)
            curr_target_hidden = target_hidden_state[:, i].contiguous()  # (B, Hidden)

            # 计算当前 Q 值
            q_vals, _ = self.agents[i](curr_obs, curr_hidden) # (B, L, n_actions)
            q_value = q_vals.gather(-1, curr_act)   # (B, L, 1)

            # 计算目标 Q 值
            with torch.no_grad():
                target_q_vals, _ = self.target_agents[i](curr_next_obs, curr_target_hidden) # (B, L, n_actions)
                max_target_q = target_q_vals.max(dim=-1, keepdim=True)[0]  # (B, L, 1)

                # 计算 TD 目标
                target = curr_rew + self.gamma * (1-curr_done) * max_target_q
            
            # 计算损失
            td_error = self.criterion(q_value, target)  # (B, L, 1)
            masked_loss = td_error * mask  # (B, L, 1)
            loss = masked_loss.sum() / mask_sum
            total_loss += loss
        
        # 计算所有智能体的平均损失
        avg_loss = total_loss / self.n_agents

        self.optimizer.zero_grad()
        avg_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agents.parameters(), 10.0)
        self.optimizer.step()

        self._train_step_count += 1
        if self._train_step_count % self.target_update_freq == 0:
            self.soft_update()
        
        return avg_loss.item()

    
    def soft_update(self):
        """ Polyak Averaging 更新 Target Network """
        for i in range(self.n_agents):
            for target_param, local_param in zip(self.target_agents[i].parameters(), self.agents[i].parameters()):
                target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path):
        torch.save(self.agents.state_dict(), path)

    def load(self, path):
        self.agents.load_state_dict(torch.load(path))
        self.target_agents.load_state_dict(self.agents.state_dict())

            