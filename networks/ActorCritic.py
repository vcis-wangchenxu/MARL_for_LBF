import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class ActorCritic(nn.Module):
    """
    Actor-Critic Network with CNN and RNN (GRU).
    Processes visual observations to produce action logits (Actor) and value estimates (Critic).
    Supports optional Agent ID embedding.
    """
    def __init__(self, obs_shape: Tuple[int, int, int], 
                 n_actions: int, 
                 n_agents: int,
                 hidden_dim: int = 64, 
                 rnn_layers: int = 1, 
                 norm_factor: float = 10.0, 
                 use_agent_id: bool = True):
        """
        Initialize the Actor-Critic network.

        Args:
            obs_shape (tuple): Observation shape (Channels, Height, Width).
            n_actions (int): Number of actions.
            n_agents (int): Number of agents (used for Agent ID embedding).
            hidden_dim (int): Dimension of the RNN hidden state.
            rnn_layers (int): Number of RNN layers.
            norm_factor (float): Normalization factor for pixel values (e.g., 255.0 or 10.0).
            use_agent_id (bool): Whether to include Agent ID in the input to RNN.
        """
        super(ActorCritic, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.rnn_layers = rnn_layers
        self.norm_factor = norm_factor
        self.use_agent_id = use_agent_id # Flag to control Agent ID usage

        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        self.cnn_out_dim = 64 * obs_shape[1] * obs_shape[2]

        # Only add n_agents to input dim if use_agent_id is True
        if self.use_agent_id:
            self.rnn_input_dim = self.cnn_out_dim + self.n_agents
        else:
            self.rnn_input_dim = self.cnn_out_dim

        self.rnn = nn.GRU(
            input_size=self.rnn_input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.rnn_layers,
            batch_first=True
        )

        self.actor = nn.Linear(self.hidden_dim, n_actions)
        self.critic = nn.Linear(self.hidden_dim, 1)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.orthogonal_(module.weight, gain=1.414)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def init_hidden(self, batch_size: int, device='cpu'):
        """
        Initialize the hidden state for the RNN.

        Args:
            batch_size (int): Batch size.
            device (str): Device to create the tensor on.

        Returns:
            torch.Tensor: Initial hidden state. Shape: (rnn_layers, batch_size, hidden_dim).
        """
        # RNN hidden state: (Num_Layers=1, Batch, Hidden)
        return torch.zeros(self.rnn_layers, batch_size, self.hidden_dim).to(device)

    def forward(self, obs, hidden_state, agent_id=None):
        """
        Forward pass of the network.

        Args:
            obs (torch.Tensor): Observations. Shape: (Batch, Seq_Len, C, H, W).
            hidden_state (torch.Tensor): RNN hidden state. Shape: (rnn_layers, Batch, hidden_dim).
            agent_id (torch.Tensor, optional): Agent IDs. Shape: (Batch, Seq_Len). Required if use_agent_id is True.

        Returns:
            logits (torch.Tensor): Action logits. Shape: (Batch, Seq_Len, n_actions).
            values (torch.Tensor): Value estimates. Shape: (Batch, Seq_Len, 1).
            new_hidden (torch.Tensor): Updated RNN hidden state. Shape: (rnn_layers, Batch, hidden_dim).
        """
        # Normalization
        obs = obs.float() / self.norm_factor

        B, L, C, H, W = obs.shape
        obs_flat = obs.reshape(B * L, C, H, W)
        
        features = self.conv(obs_flat) # (B*L, C_out, H, W)
        features = features.reshape(B * L, -1) # (B*L, cnn_out_dim)

        if self.use_agent_id:
            if agent_id is None:
                raise ValueError("Agent ID is required when use_agent_id=True")
            agent_id_flat = agent_id.reshape(-1).long()
            agent_id_one_hot = F.one_hot(agent_id_flat, num_classes=self.n_agents).float()
            # Concatenate
            rnn_in_flat = torch.cat([features, agent_id_one_hot], dim=1)
        else:
            rnn_in_flat = features

        # Reshape for RNN: (B, L, input_dim)
        rnn_in = rnn_in_flat.reshape(B, L, -1)

        self.rnn.flatten_parameters()
        rnn_out, new_hidden = self.rnn(rnn_in, hidden_state) 

        logits = self.actor(rnn_out)   # (B, L, N_Actions)
        values = self.critic(rnn_out)  # (B, L, 1)

        return logits, values, new_hidden