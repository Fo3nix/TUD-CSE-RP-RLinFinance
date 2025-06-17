# rif_reward_wrapper.py
import gymnasium as gym
import numpy as np

from common.constants import MarketDataCol


class RIFReward(gym.Wrapper):
    """
    Adds imitation-feedback r_IF and returns r_RIF = r_RF – r_IF
    to the underlying trading environment.
    The wrapped env must expose:
        • env.prices[t] – close price at t
        • env.open_prices[t+1] – open price at t+1
        • self._position – current agent position (0/1)
    """

    def __init__(self, env, oracle_labels, phi_bps: float, theta_bps: float):
        super().__init__(env)
        self.oracle = oracle_labels  # 0/1 vector for the episode
        self.phi = phi_bps / 10_000  # commission ϕ for agent
        self.theta = theta_bps / 10_000  # commission ϑ used by oracle
        self.prev_pos = 0
        self.prev_oracle = oracle_labels[0]
        self.t = 0
        market_data = env.market_data  # shape (T, C)

        self.prices = market_data[:, MarketDataCol.close_bid.value]
        self.open_prices = market_data[:, MarketDataCol.open_bid.value]

    def step(self, action):
        # -------- reinforcement feedback r_RF (Eq 5-7) :contentReference[oaicite:2]{index=2}
        trade = abs(action - self.prev_pos)
        exec_px = (self.open_prices[self.t + 1] if trade else
                   self.prices[self.t])
        r_rf = action * (self.prices[self.t + 1] - exec_px) \
               - trade * self.phi * self.open_prices[self.t + 1]

        # -------- imitation feedback r_IF  (Eq 8-9) :contentReference[oaicite:3]{index=3}
        y = self.oracle[self.t]
        trade_e = abs(y - self.prev_oracle)
        exec_px_e = (self.open_prices[self.t + 1] if trade_e else
                     self.prices[self.t])
        r_if = y * (self.prices[self.t + 1] - exec_px_e)

        # -------- combined reward  r_RIF  (Eq 10)
        reward = r_rf - r_if

        # advance wrapped env
        result = self.env.step(action)
        if len(result) == 4:
            obs, _, done, info = result
            return_obs = obs
            return_reward = reward
            return_done = done
            return_info = info
            return return_obs, return_reward, return_done, return_info
        else:
            obs, _, terminated, truncated, info = result
            return_obs = obs
            return_reward = reward
            return_terminated = terminated
            return_truncated = truncated
            return_info = info
            return return_obs, return_reward, return_terminated, return_truncated, return_info

        info.update({"r_rf": r_rf, "r_if": r_if})  # optional logging

        # book-keeping
        self.prev_pos = action
        self.prev_oracle = y
        self.t += 1
        return obs, reward, done, info

    def reset(self, **kwargs):
        self.prev_pos = 0
        self.prev_oracle = self.oracle[0]
        self.t = 0
        return self.env.reset(**kwargs)

    @property
    def episode_len(self):
        return self.env.episode_len

    @property
    def n_actions(self):
        return self.env.n_actions

    @property
    def agent_data(self):
        return self.env.agent_data
