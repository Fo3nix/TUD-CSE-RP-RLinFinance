
import argparse, yaml, os, torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from data.loader import get_ohlcv
from envs.forex_env import ForexTradingEnv
from self_rewarding.reward_network import RewardNetwork
from self_rewarding.self_reward_env import SelfRewardEnv
from utils.metrics import sharpe_ratio, sortino_ratio, cvar, max_drawdown
import pandas as pd
import wandb

def make_env(df, reward_net, expert_reward):
    def _init():
        base = ForexTradingEnv(df)
        return SelfRewardEnv(base, reward_net, expert_reward_name=expert_reward)
    return _init

def evaluate(model, df):
    env = ForexTradingEnv(df)
    obs, _ = env.reset()
    done = False
    rewards, cr = [], [1.0]
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, _, _ = env.step(action)
        rewards.append(r)
        cr.append(cr[-1]*(1 + r / env.df.loc[env.current_step-1, "close"]))
    import pandas as pd
    returns = pd.Series(rewards)
    cr_series = pd.Series(cr)
    return returns, cr_series

def summarize(returns, cr_series):
    return {
        "sharpe": sharpe_ratio(returns),
        "sortino": sortino_ratio(returns),
        "cvar": cvar(returns),
        "cum_returns": cr_series.iloc[-1],
        "max_drawdown": max_drawdown(cr_series)
    }

def train(cfg_path):
    cfg = yaml.safe_load(open(cfg_path))
    df = get_ohlcv(cfg["pair"], cfg["granularity"], cfg["start_date"], cfg["end_date"])
    input_dim = cfg.get("window_size", 50)*5
    reward_net = RewardNetwork(input_dim=input_dim, lr=cfg.get("sr_lr",1e-4))
    env = DummyVecEnv([make_env(df, reward_net, cfg["reward"]["expert_name"])])
    model = PPO(cfg["algorithm"]["policy"], env,
                learning_rate=cfg["algorithm"].get("learning_rate", 3e-4),
                seed=cfg["algorithm"]["seed"],
                verbose=1)
    model.learn(total_timesteps=cfg["algorithm"]["total_timesteps"])

    returns, cr_series = evaluate(model, df)
    metrics = summarize(returns, cr_series)
    if cfg.get("logging"):
        wandb.init(project=cfg["logging"]["project"], group="ppo_self_reward", config=cfg)
        wandb.log(metrics)
        wandb.finish()
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    train(args.config)
