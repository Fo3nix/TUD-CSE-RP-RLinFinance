import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class ForexTradingEnv(gym.Env):
    """
    Custom Environment for Forex trading compatible with Gymnasium.
    """

    def __init__(self, df, window_size=50, initial_balance=10000):
        super(ForexTradingEnv, self).__init__()

        # Forex data
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = initial_balance

        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observations: window of OHLCV + indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size, df.shape[1]),
            dtype=np.float32
        )

        self.reset()

    def reset(self, seed=None, options=None):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0  # +1 for long, -1 for short, 0 for neutral
        self.entry_price = 0
        self.total_profit = 0
        self.trades = []

        return self._get_observation(), {}

    def _get_observation(self):
        return self.df.iloc[self.current_step - self.window_size:self.current_step].values

    def step(self, action):
        done = False
        reward = 0

        price = self.df.iloc[self.current_step]['Close']

        if action == 1:  # Buy
            if self.position == 0:
                self.position = 1
                self.entry_price = price
        elif action == 2:  # Sell
            if self.position == 0:
                self.position = -1
                self.entry_price = price
        elif action == 0 and self.position != 0:
            # Close position
            if self.position == 1:
                reward = price - self.entry_price
            elif self.position == -1:
                reward = self.entry_price - price

            self.balance += reward
            self.total_profit += reward
            self.position = 0
            self.entry_price = 0
            self.trades.append(reward)

        self.current_step += 1
        if self.current_step >= len(self.df):
            done = True

        obs = self._get_observation()

        return obs, reward, done, False, {
            "balance": self.balance,
            "total_profit": self.total_profit,
            "trades": self.trades
        }

    def render(self):
        print(f'Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}')
