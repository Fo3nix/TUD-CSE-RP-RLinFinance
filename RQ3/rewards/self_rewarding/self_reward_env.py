import numpy as np
import torch
from gymnasium import Env
from envs.forex_env import ForexTradingEnv
from rewards import get_reward


class SelfRewardEnv(Env):
    """Wrap ForexTradingEnv to apply self-rewarding mechanism with a RewardNetwork."""

    def __init__(self, base_env: ForexTradingEnv, reward_net, expert_reward_name: str = "profit", device="cpu"):
        self.base_env = base_env
        self.reward_net = reward_net
        self.device = device
        self.expert_fn = get_reward(expert_reward_name)
        # mirror spaces
        self.action_space = base_env.action_space
        self.observation_space = base_env.observation_space

    def reset(self, seed=None, options=None):
        obs, info = self.base_env.reset(seed=seed, options=options)
        return obs, info

    def step(self, action):
        # run one step in base env to get raw_pnl
        obs, raw_pnl, terminated, truncated, info = self.base_env.step(action)
        # build expert reward vector (simulate if we had taken each action)
        expert_vec = np.zeros(3, dtype=np.float32)
        for a in range(3):
            expert_vec[a] = self.expert_fn(self.base_env, raw_pnl if a == action else 0.0)

        state_tensor = torch.tensor(obs.flatten(), dtype=torch.float32, device=self.device).unsqueeze(0)
        pred_vec = self.reward_net(state_tensor).detach().cpu().numpy()[0]

        reward = float(max(expert_vec[action], pred_vec[action]))

        # online update reward network
        expert_tensor = torch.tensor(expert_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.reward_net.update(state_tensor, expert_tensor)

        return obs, reward, terminated, truncated, info

    def render(self):
        self.base_env.render()
