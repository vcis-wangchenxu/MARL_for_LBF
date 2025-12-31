import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class AgentDRQN(nn.Module):
    """
    Deep Recurrent Q-Network (DRQN) agent for value-based MARL (e.g., VDN, QMIX).
    Combines CNN for visual feature extraction and GRU for handling partial observability.
    """
    def __init__(self, obs_shape: Tuple[int, int, int], 
                 n_actions: int, 
                 n_agents: int = 1,
                 rnn_hidden_dim: int = 64, 
                 rnn_layers: int = 1,
                 norm_factor: float = 10.0,
                 use_agent_id: bool = False):
        """
        Initialize the DRQN agent.

        Args:
            obs_shape (tuple): Observation shape (Channels, Height, Width).
            n_actions (int): Number of actions.
            n_agents (int): Number of agents (used for Agent ID embedding).
            rnn_hidden_dim (int): Dimension of the RNN hidden state.
            rnn_layers (int): Number of RNN layers.
            norm_factor (float): Normalization factor for pixel values.
            use_agent_id (bool): Whether to include Agent ID in the input to RNN.
        """
        super(AgentDRQN, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.norm_factor = norm_factor
        self.use_agent_id = use_agent_id

        self.conv_backbone = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.cnn_out_dim = 64 * obs_shape[1] * obs_shape[2] # 64 * 2 * 2
        
        if self.use_agent_id:
            self.rnn_input_dim = self.cnn_out_dim + self.n_agents
        else:
            self.rnn_input_dim = self.cnn_out_dim

        self.rnn = nn.GRU(
            input_size = self.rnn_input_dim,
            hidden_size = self.rnn_hidden_dim,
            num_layers = self.rnn_layers,
            batch_first = True,
        )

        self.head = nn.Linear(self.rnn_hidden_dim, n_actions)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def init_hidden(self, batch_size: int = 1, device: str = 'cpu'):
        """
        Initialize the hidden state for the RNN.

        Args:
            batch_size (int): Batch size.
            device (str): Device to create the tensor on.

        Returns:
            torch.Tensor: Initial hidden state. Shape: (rnn_layers, batch_size, rnn_hidden_dim).
        """
        return torch.zeros(self.rnn_layers, batch_size, self.rnn_hidden_dim).to(device)
    
    def forward(self, obs, hidden_state, agent_id=None):
        """
        Forward pass of the network.

        Args:
            obs (torch.Tensor): Observations. Shape: (Batch, Seq_Len, C, H, W).
            hidden_state (torch.Tensor): RNN hidden state. Shape: (rnn_layers, Batch, rnn_hidden_dim).
            agent_id (torch.Tensor, optional): Agent IDs. Shape: (Batch, Seq_Len). Required if use_agent_id is True.

        Returns:
            q_values (torch.Tensor): Q-values for each action. Shape: (Batch, Seq_Len, n_actions).
            new_hidden (torch.Tensor): Updated RNN hidden state. Shape: (rnn_layers, Batch, rnn_hidden_dim).
        """
        # Normalization
        obs = obs.float() / self.norm_factor

        B, L, C, H, W = obs.shape
        obs_flat = obs.reshape(B * L, C, H, W)
        
        # CNN Feature Extractor
        features = self.conv_backbone(obs_flat)
        features = features.reshape(B * L, -1)

        # Agent ID Injection
        if self.use_agent_id:
            if agent_id is None:
                raise ValueError("Agent ID is required for VDN/QMIX")
            agent_id_flat = agent_id.reshape(-1).long()
            agent_id_one_hot = F.one_hot(agent_id_flat, num_classes=self.n_agents).float()
            rnn_in_flat = torch.cat([features, agent_id_one_hot], dim=1)
        else:
            rnn_in_flat = features

        # RNN Sequence Processing
        rnn_in = rnn_in_flat.reshape(B, L, -1)
        
        self.rnn.flatten_parameters()
        # hidden_state shape must be (Layers, Batch, Hidden)
        rnn_out, new_hidden = self.rnn(rnn_in, hidden_state)
        
        # Q-Values
        q_values = self.head(rnn_out) # (B, L, n_actions)

        return q_values, new_hidden