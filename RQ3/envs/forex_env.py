# RQ3/envs/forex_env1.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class ForexTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self,
                 df: pd.DataFrame,
                 initial_cash: float = 10_000,
                 window_size: int = 50,
                 reward_fn=None,
                 commission: float = 0.00005,
                 dynamic_spread: bool = True):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.init_cash = initial_cash
        self.window_size = window_size
        self.reward_fn = reward_fn
        self.commission = commission
        self.dynamic_spread = dynamic_spread
        self.position = 0  # 0: flat, 1: long, -1: short
        self.entry_price = None
        self.action_space = spaces.Discrete(3)
        n_features = 6  # 6 price cols + position

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.window_size, n_features),
            dtype=np.float32)
        self.reset()

    def _obs(self):
        frame = self.df.iloc[self.step_idx - self.window_size:self.step_idx]
        price_obs = frame[['open_ask', 'high_ask', 'low_ask', 'close_ask',
                           'open_bid', 'close_bid']].values.astype(np.float32)
        price_obs += np.random.normal(0, 1e-4, price_obs.shape)
        pos_obs = np.full((self.window_size, 1), self.position, dtype=np.float32)
        return np.hstack([price_obs, pos_obs])  # shape = (window, 7)

    # 2️⃣ & 4️⃣  robust reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.position = 0  # flat
        self.entry_price = None
        self.step_idx = self.window_size
        self.history = []

        # force a random trade so PnL variance > 0
        _ = self.step(np.random.choice([1, 2]))
        return self._obs(), {}

    def _compute_pnl(self, current_price):
        if self.position == 1:  # long
            return current_price - self.entry_price - self.commission
        elif self.position == -1:  # short
            return self.entry_price - current_price - self.commission
        return 0.0

    def _mark_to_market(self, ask_price: float, bid_price: float) -> float:
        """
        Running, unrealised PnL of the *open* position.

        • long  ☞ valued at current BID  (what we could sell for)
          pnl = bid_now - entry_ask
        • short ☞ valued at current ASK  (what we must buy back for)
          pnl = entry_bid - ask_now
        • flat  ☞ 0
        """
        if self.position == 1:  # long
            return bid_price - self.entry_price
        if self.position == -1:  # short
            return self.entry_price - ask_price
        return 0.0

    def step(self, action):
        row = self.df.iloc[self.step_idx]
        ask_now = row["close_ask"]
        bid_now = row["close_bid"]

        prev_pos = self.position  # ➊ remember where we were
        reward = 0.0

        # ── update position / entry price ─────────────────────────
        if action == 1:  # go / stay long
            if prev_pos == -1:  # close short
                reward += self.entry_price - ask_now  # realise PnL
            self.position = 1
            self.entry_price = ask_now

        elif action == 2:  # go / stay short
            if prev_pos == 1:  # close long
                reward += bid_now - self.entry_price
            self.position = -1
            self.entry_price = bid_now
        # action == 0  → hold (no change)

        # if action == 3 and self.position != 0:  # close trade
        #     # realise PnL once
        #     if self.position == 1:
        #         reward += bid_now - self.entry_price
        #     else:  # short
        #         reward += self.entry_price - ask_now
        #     spread = ask_now - bid_now if self.dynamic_spread else 0.0002
        #     reward -= spread + self.commission  # pay cost ONCE
        #     self.position, self.entry_price = 0, None

        # ── charge spread/commission **only when we actually closed a trade** ──
        # if prev_pos == 0 and action in (1, 2):  # first open this trade
        #     spread = ask_now - bid_now if self.dynamic_spread else 0.0002
        #     reward -= spread + self.commission

        if self.position != 0:
            reward += self._mark_to_market(ask_now, bid_now)

        # add mark‑to‑market so reward ticks every bar
        reward += self._mark_to_market(ask_now, bid_now)

        self.history.append(reward)
        if self.reward_fn:
            reward = self.reward_fn(self, reward)

        self.step_idx += 1
        done = self.step_idx >= len(self.df) - 1
        return self._obs(), reward, done, False, {}

    def render(self):
        pass
