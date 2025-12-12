import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AgentDRQN(nn.Module):
    """
    专为 Level Based Foraging (LBF) 设计的 DRQN Agent。
    特性：
    1. 动态适应任意 sight 范围 (通过 AdaptiveAvgPool2d)。
    2. 内部自动归一化 (输入 0-10 整数，内部转为 0-1 浮点数)。
    3. 同时支持训练时的序列输入 (Sequence) 和推理时的单步输入 (Step)。
    """
    def __init__(self, obs_shape: Tuple[int, int, int], 
                 n_actions: int, rnn_hidden_dim: int = 64, 
                 norm_factor: float = 10.0):
        """
        Parameters:
        - obs_shape: (C, H, W) 观测维度，例如 (3, 5, 5) 或 (3, 17, 17)
        - n_actions: 动作空间大小
        - rnn_hidden_dim: GRU 隐藏层维度
        - norm_factor: 归一化系数，通常设为环境中的最大等级 (例如 10.0)
        """
        super(AgentDRQN, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.rnn_hidden_dim = rnn_hidden_dim
        self.norm_factor = norm_factor

        self.conv_backbone = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.cnn_out_dim = 64 * 2 * 2

        self.rnn = nn.GRU(
            input_size = self.cnn_out_dim,
            hidden_size = self.rnn_hidden_dim,
            num_layers = 1,
            batch_first = False,
        )

        self.fc_q = nn.Linear(self.rnn_hidden_dim, self.n_actions)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """Return a zero hidden state tensor shaped (batch_size, hidden_dim)."""
        return torch.zeros(batch_size, self.rnn_hidden_dim, device=device)
    
    def forward(self, obs_input: torch.Tensor, hidden_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        obs_input_norm = obs_input.float() / self.norm_factor
        
        if obs_input.dim() == 5:
            is_sequence = True
            B, L, C, H, W = obs_input_norm.shape
            obs_flat = obs_input_norm.reshape(-1, C, H, W)    # (B*L, C, H, W)
        else:
            is_sequence = False
            B, C, H, W = obs_input_norm.shape                 # (B, C, H, W)
            L = 1
            obs_flat = obs_input_norm                         # (B, C, H, W)
        
        features = self.conv_backbone(obs_flat)  # (B*L, Feat_C, Feat_H, Feat_W)
        features = self.adaptive_pool(features)  # (B*L, Feat_C, 2, 2)
        cnn_out = features.flatten(start_dim=1)  # (B*L, cnn_out_dim)

        # 先把 B*L 拆回 (B, L)，再转置为 (L, B) 供 RNN 使用
        rnn_input = cnn_out.view(B, L, self.cnn_out_dim).permute(1, 0, 2) # (L, B, H)

        h_in = hidden_state.reshape(1, B, self.rnn_hidden_dim)  # (1, B, rnn_hidden_dim)
        rnn_output, h_n = self.rnn(rnn_input, h_in)  # rnn_output: (L, B, rnn_hidden_dim)

        # 同样需要先转置回 (B, L) 再展平，保证数据顺序一致
        rnn_output_batch_first = rnn_output.permute(1, 0, 2) # (B, L, H)
        q_flat = self.fc_q(rnn_output_batch_first.reshape(-1, self.rnn_hidden_dim)) # (B*L, n_actions)

        if is_sequence:
            q_values = q_flat.reshape(B, L, self.n_actions)  # (B, L, n_actions)
        else:
            q_values = q_flat.reshape(B, self.n_actions)     # (B, n_actions)

        next_hidden = h_n.reshape(B, self.rnn_hidden_dim)
        return q_values, next_hidden
