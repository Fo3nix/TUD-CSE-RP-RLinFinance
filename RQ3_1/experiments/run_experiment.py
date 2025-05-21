# RQ3/experiments/run_experiment.py
"""
Generic RL‑experiment runner for the RQ3 codebase.

Supports any reward function defined in RQ3.rewards,
with fully parameterized configuration via YAML.
"""

import argparse
from collections import Counter

import yaml

import pandas as pd
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from RQ3.envs.forex_env import ForexTradingEnv
from RQ3.rewards.global_history import GLOBAL_HISTORY
from RQ3.rewards.get_reward import get_reward
from RQ3.utils.metrics import (sharpe_ratio,
                               sortino_ratio,
                               cvar,
                               max_drawdown)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ─────────────────────────── data loader ───────────────────────────────────────
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


# ─────────────────────────── training loop ─────────────────────────────────────
def reset_weights(policy):
    for layer in policy.mlp_extractor.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.reset_parameters()
    for layer in policy.action_net.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.reset_parameters()
    for layer in policy.value_net.modules():
        if isinstance(layer, (nn.Linear, nn.Conv2d)):
            layer.reset_parameters()
    print("[DEBUG] Policy weights have been reset.")

def train(cfg):
    # Load and split data
    train_df, _ = load_csv_bid_ask(
        cfg["data"]["ask_csv"],
        cfg["data"]["bid_csv"],
        train_start=cfg["data"].get("train_start"),
        train_end=cfg["data"].get("train_end")
    )

    # Clear history at the start of each training run
    GLOBAL_HISTORY.reset()

    # Build the reward function
    reward_fn = get_reward(cfg["reward"]["name"])
    reward_params = cfg["reward"].get("params", {})

    # Build the environment
    env_fn = lambda: ForexTradingEnv(
        train_df,
        reward_fn=lambda env, pnl: reward_fn(env, pnl, **reward_params)
    )

    env = DummyVecEnv([env_fn])

    # Build PPO agent
    model = PPO(cfg["algo"]["policy"],
                env,
                learning_rate=cfg["algo"].get("learning_rate", 3e-4),
                seed=cfg["algo"].get("seed", 42),
                ent_coef=cfg["algo"].get("ent_coef", 0.01),
                verbose=1)

    reset_weights(model.policy)

    # Train the model
    model.learn(total_timesteps=cfg["algo"]["total_timesteps"])

    return model, train_df


actions = []


# def evaluate(model, df):
#     env = ForexTradingEnv(df)
#     obs, _ = env.reset()
#     done = False
#     rewards, equity = [], [1.0]
#     position_entry_price = None
#     position_direction = 0  # 0: flat, 1: long, -1: short
#
#     while not done:
#         action, _ = model.predict(obs, deterministic=True)
#         actions.append(action)
#         prev_price = position_entry_price or (
#                     (env.df.loc[env.step_idx - 1, "close_ask"] + env.df.loc[env.step_idx - 1, "close_bid"]) / 2)
#         obs, reward, done, _, _ = env.step(action)
#
#         # Update position
#         if action == 1:  # long
#             position_direction = 1
#             position_entry_price = env.df.loc[env.step_idx - 1, "close_ask"]
#         elif action == 2:  # short
#             position_direction = -1
#             position_entry_price = env.df.loc[env.step_idx - 1, "close_bid"]
#
#         # Calculate PnL based on actual fills
#         if position_direction == 1:  # long
#             exit_price = env.df.loc[env.step_idx - 1, "close_bid"]
#         elif position_direction == -1:  # short
#             exit_price = env.df.loc[env.step_idx - 1, "close_ask"]
#         else:
#             exit_price = prev_price  # no position, use last mid‑price
#
#         equity.append(equity[-1] * (1 + reward / exit_price))
#         rewards.append(reward)
#
#     print("First 50 actions:", actions[:50])
#     returns = pd.Series(rewards)
#     equity = pd.Series(equity)
#
#     return {
#         "sharpe": sharpe_ratio(returns),
#         "sortino": sortino_ratio(returns),
#         "cvar": cvar(returns),
#         "cum_returns": equity.iloc[-1],
#         "max_drawdown": max_drawdown(equity),
#     }

def evaluate(model, df):
    env = ForexTradingEnv(df)
    obs, _ = env.reset()
    done = False
    actions = []
    rewards, equity = [], [1.0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(int(action))
        obs, reward, done, _, _ = env.step(action)
        equity.append(equity[-1] * (1 + reward / max(1e-8, equity[-1])))
        rewards.append(reward)

    returns = pd.Series(rewards)
    equity = pd.Series(equity)

    print("First 50 actions:", actions[:50])
    print("Action distribution:", Counter(actions))

    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "cvar": cvar(returns),
        "cum_returns": equity.iloc[-1],
        "max_drawdown": max_drawdown(equity),
    }


# ─────────────────────────── entry‑point / CLI ───────────────────────────────
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True, help="Path to YAML config")
#     args = parser.parse_args()
#
#     # Load the full YAML config
#     with open(args.config, "r") as f:
#         cfg = yaml.safe_load(f)
#
#     # Train
#     model, df = train(cfg)
#
#     # Evaluate
#     metrics = evaluate(model, df)
#
#     # Log
#     if WANDB_AVAILABLE:
#         wandb.init(project=cfg["logging"]["project"], group=cfg["logging"]["group"], config=cfg)
#         wandb.log(metrics)
#         wandb.finish()
#
#     print("\nResults for reward =", cfg["reward"]["name"])
#     for k, v in metrics.items():
#         print(f"{k:12s}: {v:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()

    # Load the full YAML config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Train the model
    model, train_df = train(cfg)

    # Load test data
    _, test_df = load_csv_bid_ask(
        cfg["data"]["ask_csv"],
        cfg["data"]["bid_csv"],
        test_start=cfg["data"].get("test_start"),
        test_end=cfg["data"].get("test_end")
    )

    # Evaluate on the test set
    metrics = evaluate(model, test_df)

    print("\nResults for reward =", cfg["reward"]["name"])
    for k, v in metrics.items():
        print(f"{k:12s}: {v:.4f}")



if __name__ == "__main__":
    main()
