# forex_dqn.py
"""Forex DQN Trading Environment for EUR/USD

This module defines a custom Gymnasium environment for training a
Deep Q‑Network (DQN) via Stable‑Baselines3 on the EUR/USD forex pair.

Features
--------
* Windowed observations of price‑based and technical indicators.
* Discrete actions: 0=Hold, 1=Long, 2=Short.
* Simple wealth‑based reward with transaction costs.
* Two position‑sizing modes:
    - Fixed fraction of equity.
    - Volatility‑targeting (optional).

Quick start
-----------
$ pip install gymnasium stable-baselines3 pandas numpy ta
$ python forex_dqn.py --train /Users/justasbertasius/PycharmProjects/TUD-CSE-RP-RLinFinance(1)/data/EURUSD_Candlestick_1_Hour_ASK_01.01.2023-01.01.2025.csv

"""

from __future__ import annotations
import argparse
import os
import pathlib
from collections import Counter
from typing import Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces

import numpy as np
import pandas as pd
import ta  # technical analysis helpers
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ3.utils.metrics import sharpe_ratio, sortino_ratio, cvar, max_drawdown

from stable_baselines3.common.callbacks import BaseCallback


class TrainingRewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_rewards = []

    def _on_step(self) -> bool:
        reward = self.locals['rewards'][0]  # SB3 uses vectorized envs
        self.current_rewards.append(reward)
        done = self.locals['dones'][0]
        if done:
            self.episode_rewards.append(np.sum(self.current_rewards))
            self.current_rewards = []
        return True


def load_csv_bid_ask(ask_csv: str, train_start=None, train_end=None, test_start=None, test_end=None):
    # Load and harmonize column names
    ask = pd.read_csv(ask_csv)

    # Convert timestamps
    ask.rename(columns={'Gmt time': 'gmt_time'}, inplace=True)
    ask['gmt_time'] = pd.to_datetime(ask['gmt_time'], dayfirst=True)
    # Merge on timestamp
    df = ask
    df.rename(columns={"gmt_time": "time"}, inplace=True)
    df.sort_values("time", inplace=True)

    # Apply train/test split
    if train_start and train_end:
        train_df = df[
            (df['time'] >= pd.to_datetime(train_start)) & (df['time'] <= pd.to_datetime(train_end))].reset_index(
            drop=True)
    else:
        train_df = df

    if test_start and test_end:
        test_df = df[(df['time'] >= pd.to_datetime(test_start)) & (df['time'] <= pd.to_datetime(test_end))].reset_index(
            drop=True)
    else:
        test_df = df

    return train_df, test_df


plot_dir = "/Users/justasbertasius/PycharmProjects/TUD-CSE-RP-RLinFinance(1)/RQ3_1/results"


def save_equity_curve(equity: pd.Series, filename="equity_curve_save.png", title="Equity Curve"):
    plt.figure(figsize=(10, 6))
    plt.plot(equity.index, equity.values, label="Equity", color='dodgerblue')
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_action_distribution(actions: list, filename="action_distribution_save.png"):
    action_counts = Counter(actions)
    labels = ['Hold', 'Long', 'Short']
    values = [action_counts.get(0, 0), action_counts.get(1, 0), action_counts.get(2, 0)]

    plt.figure(figsize=(7, 5))
    plt.bar(labels, values, color='mediumseagreen')
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def save_return_distribution(rewards: list, filename="return_distribution.png"):
    plt.figure(figsize=(9, 5))
    plt.hist(rewards, bins=100, alpha=0.75, color='salmon')
    plt.title("Return Distribution")
    plt.xlabel("Reward/Return")
    plt.ylabel("Frequency")
    path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def make_metrics_table(metrics: dict, action_counts: dict) -> pd.DataFrame:
    data = metrics.copy()
    data.update({
        "n_hold": action_counts.get(0, 0),
        "n_long": action_counts.get(1, 0),
        "n_short": action_counts.get(2, 0),
        "total_actions": sum(action_counts.values())
    })
    df = pd.DataFrame([data])
    return df


def save_metrics_table_as_image(metrics, action_counts, filename="metrics_table.png"):
    df = make_metrics_table(metrics, action_counts)
    fig, ax = plt.subplots(figsize=(10, 1.5 + 0.25 * len(df.columns)))
    ax.axis('off')
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     loc='center',
                     cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)
    path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def save_training_return_curve(training_rewards: list, filename="training_return_curve.png"):
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    cum_returns = np.cumprod([1 + r for r in training_rewards])
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(cum_returns)), cum_returns, color="mediumblue", label="Cumulative Return")
    plt.title("Cumulative Return During Training")
    plt.xlabel("Training Step")
    plt.ylabel("Cumulative Return")
    plt.grid(True)
    plt.legend()
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def _compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators used as state features."""
    # Price returns
    df['ret_1h'] = df['close'].pct_change()
    for window in [6, 12, 24, 168]:  # 6h,12h,24h,7d
        df[f'ret_{window}h'] = df['close'].pct_change(window)
    # MACD (fast=12, slow=26, signal=9 by convention)
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd_diff()
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    # Rolling volatility (std of hourly returns annualised)
    df['vol_24h'] = df['ret_1h'].rolling(24).std() * np.sqrt(24 * 252)
    df.dropna(inplace=True)
    return df


class ForexTradingEnv(gym.Env):
    """A minimal EUR/USD trading env with DQN‑friendly API."""

    metadata = {'render.modes': ['human']}

    def __init__(
            self,
            df: pd.DataFrame,
            window_size: int = 24,
            initial_balance: float = 10000,
            pos_fraction: float = 0.1,
            volatility_target: float = 0.15,
            transaction_cost: float = 0.00005,  # 0.5 bp
            mode: str = 'vol_target'  # 'fixed' or 'vol_target'
    ):
        super().__init__()
        assert mode in {'fixed', 'vol_target'}, "Invalid position sizing mode"
        self.df = df.reset_index(drop=True)
        self.window_size = window_size
        self.initial_balance = float(initial_balance)
        self.pos_fraction = pos_fraction
        self.vol_target = volatility_target
        self.tc = transaction_cost
        self.mode = mode

        # Features used in observations
        self.feature_columns = [
            'close', 'ret_1h', 'ret_6h', 'ret_12h', 'ret_24h', 'ret_168h',
            'macd', 'rsi', 'vol_24h'
        ]
        # Build spaces
        obs_len = window_size * len(self.feature_columns)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32
        )

        self._reset_internal_state()

    # ------------------------------------------------------------------ #
    #  Core RL API                                                       #
    # ------------------------------------------------------------------ #
    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        super().reset(seed=seed)
        self._reset_internal_state()
        return self._get_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        if self._done:
            raise RuntimeError("Episode finished – call reset()")

        # Advance time index
        self.current_step += 1
        if self.current_step >= len(self.df) - 1:
            self._done = True

        price_prev = self.df.loc[self.current_step - 1, 'close']
        price_now = self.df.loc[self.current_step, 'close']
        ret = (price_now / price_prev) - 1.0

        # Position sizing
        if action == 0:  # Hold
            target_position = self.position
        else:
            sign = 1 if action == 1 else -1
            if self.mode == 'fixed':
                target_position = sign * self.pos_fraction * self.balance
            else:
                vol_est = self.df.loc[self.current_step - 1, 'vol_24h']
                scaling = self.vol_target / max(vol_est, 1e-6)
                target_position = sign * scaling * self.balance

        # Transaction costs for changes in position
        trade_amount = target_position - self.position
        cost = abs(trade_amount) * self.tc
        self.balance -= cost

        # Mark‑to‑market PnL
        pnl = self.position * ret
        self.balance += pnl
        self.position = target_position

        self.net_worth = self.balance + self.position
        reward = (self.net_worth - self.prev_worth) / self.prev_worth
        self.prev_worth = self.net_worth

        # Liquidation condition
        if self.net_worth <= self.initial_balance * 0.5:
            self._done = True

        return self._get_observation(), float(reward), self._done, False, {}

    def render(self, mode: str = 'human'):
        print(f"Step: {self.current_step}  NetWorth: {self.net_worth:,.2f}")

    # ------------------------------------------------------------------ #
    #  Helpers                                                           #
    # ------------------------------------------------------------------ #
    def _reset_internal_state(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.position = 0.0
        self.net_worth = self.initial_balance
        self.prev_worth = self.net_worth
        self._done = False

    def _get_observation(self) -> np.ndarray:
        frame = self.df.loc[self.current_step - self.window_size: self.current_step - 1,
                self.feature_columns]
        obs = frame.to_numpy().flatten().astype(np.float32)
        return obs


# ---------------------------------------------------------------------- #
#  Training / evaluation helpers                                         #
# ---------------------------------------------------------------------- #

def make_env(df: pd.DataFrame, **kwargs):
    return lambda: ForexTradingEnv(df, **kwargs)


def train(df, timesteps: int = 200_000, **env_kwargs):
    df.columns = df.columns.str.lower()
    df = _compute_features(df)
    env = DummyVecEnv([make_env(df, **env_kwargs)])

    model = DQN(
        "MlpPolicy", env,
        learning_rate=1e-4,
        buffer_size=50_000,
        batch_size=32,
        gamma=0.99,
        exploration_fraction=0.1,
        target_update_interval=1_000,
        verbose=1
    )
    reward_callback = TrainingRewardCallback()
    model.learn(total_timesteps=timesteps, callback=reward_callback)
    model.save("dqn_eurusd_1")

    # Plot cumulative return from rewards
    episode_returns = reward_callback.episode_rewards
    training_rewards = np.repeat(episode_returns, int(timesteps / len(episode_returns)))[:timesteps]
    save_training_return_curve(training_rewards)
    return model


def evaluate(model, df, **env_kwargs):
    df.columns = df.columns.str.lower()
    df = _compute_features(df)
    env = DummyVecEnv([make_env(df, **env_kwargs)])
    obs = env.reset()
    done = False
    total_return = 0.0
    actions = []
    rewards, equity = [], [1.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        action_s = int(action.item())  # safer extraction
        actions.append(action_s)
        obs, reward, done, _ = env.step(action)
        total_return += reward
        equity.append(equity[-1] * (1 + reward / max(1e-8, equity[-1])))
        rewards.append(reward.item())

    returns = pd.Series(rewards)
    equity = pd.Series(equity)
    action_counts = Counter(actions)
    print(f"Strategy return: {total_return.item() * 100:.2f}%")
    print("Action distribution:", action_counts)

    metrics = {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "cvar": cvar(returns),
        "cum_returns": equity.iloc[-1],
        "max_drawdown": max_drawdown(equity),
    }

    print(metrics)

    save_equity_curve(equity)
    save_action_distribution(actions)
    save_return_distribution(rewards)
    save_metrics_table_as_image(metrics, action_counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train')
    parser.add_argument('--timesteps', type=int, default=200_000)
    parser.add_argument('--evaluate', type=str, help='Path to trained model ZIP')
    args = parser.parse_args()

    train_df, test_df = load_csv_bid_ask(
        '/Users/justasbertasius/PycharmProjects/TUD-CSE-RP-RLinFinance(1)/data/EURUSD_Candlestick_1_Hour_ASK_01.01.2023-01.01.2025.csv',
        train_start="2023-01-01",
        train_end="2024-07-31",
        test_start="2024-08-01",
        test_end="2025-01-01")

    model = train(
        train_df,
        timesteps=100000)

    evaluate(model,
             test_df)
