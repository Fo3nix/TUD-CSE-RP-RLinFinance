# RQ3_1/experiments/run_dqn_stockstats.py
"""Train and evaluate a DQN trader on the new StockStats‑powered ForexEnv.

Usage
-----
python -m RQ3_1.experiments.run_dqn_stockstats --config RQ3_1/configs/baseline.yaml
"""
import argparse
from pathlib import Path
from collections import Counter
import yaml
import numpy as np
import pandas as pd

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------------------------------------------------------------------
#  Local import: the new environment lives in RQ3_1/envs/forex_env_stockstats.py
# ---------------------------------------------------------------------------
from RQ3_1.envs.forex_env2 import ForexTradingEnv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv_bid_ask(ask_csv: str, bid_csv: str, start=None, end=None):
    """Load ASK & BID csvs and return a single DataFrame expected by ForexEnv.
    The resulting df has bid/ask OHLC columns and optional volume.
    """
    ask = pd.read_csv(ask_csv)
    bid = pd.read_csv(bid_csv)
    ask.columns = bid.columns = [c.lower().replace(' ', '_') for c in ask.columns]
    ask['gmt_time'] = pd.to_datetime(ask['gmt_time'], dayfirst=True)
    bid['gmt_time'] = pd.to_datetime(bid['gmt_time'], dayfirst=True)

    df = pd.merge(ask, bid, on='gmt_time', suffixes=('_ask', '_bid'))
    df.rename(columns={'gmt_time': 'timestamp'}, inplace=True)
    df.sort_values('timestamp', inplace=True)

    if start:
        df = df[df['timestamp'] >= pd.to_datetime(start)]
    if end:
        df = df[df['timestamp'] <= pd.to_datetime(end)]

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def make_env(df, cfg, seed):
    return ForexTradingEnv(
        df,
        # initial_capital=cfg['env'].get('initial_capital', 10_000),
        # transaction_cost_pct=cfg['env'].get('transaction_cost_pct', 0.0),
        # lookback_window_size=cfg['env'].get('window_size', 30),
        # log_level=cfg['env'].get('log_level', 1),
        # seed=seed,
    )


def train(cfg):
    df = load_csv_bid_ask(cfg['data']['ask_csv'], cfg['data']['bid_csv'],
                          start=cfg['data'].get('train_start'),
                          end=cfg['data'].get('train_end'))

    env = DummyVecEnv([lambda: make_env(df, cfg, seed=42)])

    policy_kwargs = dict(net_arch=cfg['algo'].get('net_arch', [128, 128]))

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=cfg['algo'].get('learning_rate', 1e-3),
        buffer_size=cfg['algo'].get('buffer_size', 50_000),
        learning_starts=cfg['algo'].get('learning_starts', 1_000),
        batch_size=cfg['algo'].get('batch_size', 64),
        gamma=cfg['algo'].get('gamma', 0.99),
        tau=cfg['algo'].get('tau', 1.0),
        train_freq=cfg['algo'].get('train_freq', 4),
        target_update_interval=cfg['algo'].get('target_update_interval', 500),
        exploration_fraction=cfg['algo'].get('exploration_fraction', 0.2),
        exploration_final_eps=cfg['algo'].get('exploration_final_eps', 0.05),
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=cfg['algo'].get('seed', 42),
    )

    model.learn(total_timesteps=cfg['algo']['total_timesteps'], log_interval=100)
    return model


def evaluate(model, cfg):
    test_df = load_csv_bid_ask(cfg['data']['ask_csv'], cfg['data']['bid_csv'],
                               start=cfg['data'].get('test_start'),
                               end=cfg['data'].get('test_end'))

    eval_env = DummyVecEnv([lambda: make_env(test_df, cfg, seed=123)])

    obs = eval_env.reset()
    done = False
    total_reward = 0
    actions = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)

        reward = float(reward[0])  # for SB3's DummyVecEnv
        action = int(action[0])

        actions.append(action)
        total_reward += reward

    print("Action distribution:", Counter(actions))
    print("Total reward on eval:", total_reward)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    model = train(cfg)
    evaluate(model, cfg)
