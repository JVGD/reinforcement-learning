import torch as T
from torch import nn

class QNetwork(nn.Module):
    def __init__(self, N_states, N_actions, hidden=10):
        # Init torch module
        super(QNetwork, self).__init__()

        # QNetwork
        self.Q_net = nn.ModuleList([
            nn.Linear(N_states, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, N_actions, bias=True),
        ])

    def forward(self, x):
        # Simple forward call
        for layer in self.Q_net:
            x = layer(x)
        return x