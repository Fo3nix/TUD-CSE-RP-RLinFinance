
import torch
import torch.nn as nn
import torch.optim as optim

class RewardNetwork(nn.Module):
    """Simple MLP that predicts a reward value for each discrete action."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, lr: float = 1e-4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)
        )
        self.opt = optim.Adam(self.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def update(self, states, expert_rewards):
        self.train()
        preds = self(states)
        loss = self.loss_fn(preds, expert_rewards)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
