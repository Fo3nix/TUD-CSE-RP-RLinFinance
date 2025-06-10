# --------------------------------------------------------------------- #
#  RewardNet  — selects NLinear / WFTNet / TimesNet at runtime
# --------------------------------------------------------------------- #
import torch
import torch.nn as nn
from typing import Literal
from typing import Callable
import gymnasium as gym

# Optional: pip install timesnet-pytorch  wftnet-pytorch  nlinear-pytorch
try:
    from timesnet_pytorch import TimesNet
    from wftnet_pytorch import WFTNet
    from nlinear_pytorch import NLinear
except ImportError:
    TimesNet = WFTNet = NLinear = None  # graceful fallback


class RewardNet(nn.Module):
    """
    Reward network used in SRDDQN (cf. Fig. 2 in the paper).
    Feature-extraction block = {TimesNet, WFTNet, NLinear}
    Reward-head = 2-layer MLP → n_actions.
    """

    def __init__(
            self,
            obs_dim: int,
            n_actions: int,
            model_type: Literal["timesnet", "wftnet", "nlinear", "mlp"] = "mlp",
            hidden: tuple[int, int] = (128, 64)
    ):
        super().__init__()

        # 1)  choose backbone
        if model_type == "timesnet":
            assert TimesNet, "pip install timesnet-pytorch"
            self.backbone = TimesNet(input_size=obs_dim, target_size=hidden[0])
        elif model_type == "wftnet":
            assert WFTNet, "pip install wftnet-pytorch"
            self.backbone = WFTNet(input_size=obs_dim, target_size=hidden[0])
        elif model_type == "nlinear":
            assert NLinear, "pip install nlinear-pytorch"
            self.backbone = NLinear(input_size=obs_dim, target_size=hidden[0])
        elif model_type == "mlp":
            self.backbone = nn.Sequential(
                nn.Linear(obs_dim, hidden[0]),
                nn.GELU()  # simple non-linearity
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # 2)  small MLP head (Table 1 lists 2-MLP for all three nets):contentReference[oaicite:1]{index=1}
        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden[0], hidden[1]),
            nn.GELU(),
            nn.Linear(hidden[1], n_actions)
        )

    def forward(self, x):
        # x shape: (B, obs_dim)
        feats = self.backbone(x.unsqueeze(1))  # many libs expect (B, C=1, L)
        return self.head(feats.squeeze(1))  # (B, n_actions)

class SelfRewardingEnv(gym.Wrapper):
    """
    Wraps any Gym-style trading env and implements the
    SRDRL mechanism (Eq. 8–9 in the paper).
    """
    def __init__(
        self,
        env: gym.Env,
        reward_net: nn.Module,
        expert_reward_fn: Callable[[gym.Env], torch.Tensor],
        lr: float = 1e-4,
        device: str = "cpu"
    ):
        super().__init__(env)
        self.device = device
        self.reward_net = reward_net.to(device)
        self.optim = torch.optim.Adam(self.reward_net.parameters(), lr=lr)
        self.mse = nn.MSELoss()
        self.expert_reward_fn = expert_reward_fn          # returns Tensor (n_actions,)

    # ---------- main hook ----------
    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)

        # 1)  build per-action expert reward   (vector)
        r_exp = self.expert_reward_fn(self.env).to(self.device)      # (n_actions,)

        # 2)  predict rewards with RewardNet
        obs_t = torch.as_tensor(obs, dtype=torch.float32,
                                device=self.device).unsqueeze(0)     # (1, obs_dim)
        r_pred = self.reward_net(obs_t).squeeze(0).detach()          # (n_actions,)

        # 3)  choose the higher reward for each action (Eq. 9)
        r_best = torch.maximum(r_pred, r_exp)

        # 4)  scalar reward fed to the RL agent
        reward = r_best[action].item()

        # 5)  online update of RewardNet (MSE to r_best)
        if not terminated and not truncated:
            pred_for_loss = self.reward_net(obs_t).squeeze(0)        # forward again (grad on)
            loss = self.mse(pred_for_loss, r_best)
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        return obs, reward, terminated, truncated, info

    # gym reset passthrough
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return obs, info

    def __getattr__(self, name):
        return getattr(self.env, name)
