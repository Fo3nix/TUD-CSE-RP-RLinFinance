# RQ3/experiments/run_dqn_experiment.py
"""Train and evaluate a DQN trader (discrete‑action) on the fixed ForexTradingEnv.

Usage
-----
python -m RQ3.experiments.run_dqn_experiment --config RQ3/configs/dqn_baseline.yaml
"""

import argparse
import yaml
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback

from RQ3.envs.forex_env import ForexTradingEnv
from RQ3.rewards.get_reward import get_reward
from RQ3.rewards.global_history import GLOBAL_HISTORY
from RQ3.utils.metrics import sharpe_ratio, sortino_ratio, cvar, max_drawdown
from RQ3_1.envs.forex_env1 import ForexEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_csv_ask_only(ask_csv: str, train_start=None, train_end=None, test_start=None, test_end=None):
    # Load ASK CSV and standardize columns
    ask = pd.read_csv(ask_csv)
    ask.columns = [c.lower().replace(' ', '_') for c in ask.columns]
    ask['gmt_time'] = pd.to_datetime(ask['gmt_time'], dayfirst=True)
    ask.rename(columns={'gmt_time': 'time'}, inplace=True)
    ask.sort_values("time", inplace=True)

    # Rename close column to match expected 'Close' for the environment
    if 'close' not in ask.columns:
        raise ValueError("Expected a 'close' column in the ASK CSV.")

    ask['Close'] = ask['close']  # Capital 'C' for compatibility
    ask.reset_index(drop=True, inplace=True)

    # Train/Test splitting
    if train_start and train_end:
        train_df = ask[
            (ask['time'] >= pd.to_datetime(train_start)) & (ask['time'] <= pd.to_datetime(train_end))
        ].reset_index(drop=True)
    else:
        train_df = ask

    if test_start and test_end:
        test_df = ask[
            (ask['time'] >= pd.to_datetime(test_start)) & (ask['time'] <= pd.to_datetime(test_end))
        ].reset_index(drop=True)
    else:
        test_df = ask

    return train_df, test_df


def load_csv_bid_ask(ask_csv: str, bid_csv: str, train_start=None, train_end=None, test_start=None, test_end=None):
    # Load and harmonize column names
    ask = pd.read_csv(ask_csv)
    bid = pd.read_csv(bid_csv)

    ask.columns = bid.columns = [c.lower().replace(' ', '_') for c in ask.columns]

    # Convert timestamps
    ask['gmt_time'] = pd.to_datetime(ask['gmt_time'], dayfirst=True)
    bid['gmt_time'] = pd.to_datetime(bid['gmt_time'], dayfirst=True)

    # Merge on timestamp
    df = pd.merge(ask, bid, on="gmt_time", suffixes=("_ask", "_bid"))
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


# def load_csv_bid_ask(ask_csv: str, bid_csv: str, *, start=None, end=None):
#     """Merge separate ASK/BID csvs into a single DataFrame."""
#     ask = pd.read_csv(ask_csv)
#     bid = pd.read_csv(bid_csv)
#     # normalise column names -> lower snake
#     ask.columns = bid.columns = [c.lower().replace(" ", "_") for c in ask.columns]
#     ask['gmt_time'] = pd.to_datetime(ask['gmt_time'], dayfirst=True)
#     bid['gmt_time'] = pd.to_datetime(bid['gmt_time'], dayfirst=True)
#     df = pd.merge(ask, bid, on='gmt_time', suffixes=('_ask', '_bid'))
#     df.rename(columns={'gmt_time': 'time'}, inplace=True)
#     if start:
#         df = df[df['time'] >= pd.to_datetime(start)]
#     if end:
#         df = df[df['time'] <= pd.to_datetime(end)]
#     return df.reset_index(drop=True)

# ---------------------------------------------------------------------------
# Training + evaluation
# ---------------------------------------------------------------------------

def train(cfg):
    # 1) data split ----------------------------------------------------------
    train_df, _ = load_csv_ask_only(cfg['data']['ask_csv'],
                                    train_start=cfg['data'].get('train_start'),
                                    train_end=cfg['data'].get('train_end'))

    # 2) reward fn -----------------------------------------------------------
    GLOBAL_HISTORY.reset()
    reward_fn = get_reward(cfg['reward']['name'])
    reward_params = cfg['reward'].get('params', {})

    def make_env():
        return ForexEnv(
            train_df,
            window_size=10
            # reward_fn=lambda env, pnl: reward_fn(env, pnl, **reward_params),
            # commission=cfg['env'].get('commission', 0.00005),
            # dynamic_spread=cfg['env'].get('dynamic_spread', True),
            # window_size=cfg['env'].get('window_size', 50),
        )

    env = DummyVecEnv([make_env])

    # 3) model ---------------------------------------------------------------
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg['algo'].get('learning_rate', 1e-4),
        buffer_size=cfg['algo'].get('buffer_size', 100_000),
        batch_size=cfg['algo'].get('batch_size', 256),
        tau=cfg['algo'].get('tau', 0.005),
        gamma=cfg['algo'].get('gamma', 0.99),
        exploration_fraction=cfg['algo'].get('exploration_fraction', 0.1),
        exploration_final_eps=cfg['algo'].get('exploration_final_eps', 0.02),
        target_update_interval=cfg['algo'].get('target_update_interval', 10_000),
        seed=cfg['algo'].get('seed', 42),
        verbose=1,
    )

    model.learn(total_timesteps=cfg['algo']['total_timesteps'])
    return model


def evaluate(model, cfg):
    _, test_df = load_csv_ask_only(cfg['data']['ask_csv'],
                                   test_start=cfg['data'].get('test_start'),
                                   test_end=cfg['data'].get('test_end'))

    env = ForexEnv(test_df, window_size=10)
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
    assert obs.shape == model.observation_space.shape, f"Expected {model.observation_space.shape}, got {obs.shape}"
    done = False
    actions, rewards, equity = [], [], [1.0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, r, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        rewards.append(r)
        equity.append(equity[-1] + r)  # additive equity curve

    print("Action distribution:", Counter(actions))

    returns = pd.Series(rewards)
    eq = pd.Series(equity)
    return {
        'sharpe': sharpe_ratio(returns),
        'sortino': sortino_ratio(returns),
        'cvar': cvar(returns),
        'cum_returns': eq.iloc[-1],
        'max_drawdown': max_drawdown(eq),
    }


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    model = train(cfg)
    metrics = evaluate(model, cfg)

    print("\nResults for reward =", cfg['reward']['name'])
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")


if __name__ == '__main__':
    main()
