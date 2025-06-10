import logging
from datetime import datetime

from stable_baselines3 import DQN, A2C
from torch import nn, optim
from torch.nn import ReLU, LeakyReLU

from RQ3.self_rewarding.expert_rewards import expert_reward_vector
from RQ3.self_rewarding.reward_net import RewardNet, SelfRewardingEnv
from common.data.data import ForexCandleData, Timeframe
from common.data.feature_engineer import (FeatureEngineer, adx,
                                          as_min_max_fixed, as_min_max_window,
                                          as_pct_change,
                                          as_ratio_of_other_column, as_z_score,
                                          atr, bollinger_bands, cci,
                                          copy_column, ema,
                                          historic_pct_change, macd, rsi,
                                          stochastic_oscillator)
from common.data.stepwise_feature_engineer import (StepwiseFeatureEngineer,
                                                   calculate_current_exposure)
from common.envs.forex_env import ForexEnv
from RQ3.profit_based import (equity_change, log_equity_change)
from RQ3.risk_adjusted import volatility_scaled_reward, risk_adjusted_return, sharpe_ratio_reward, cvar_reward
from RQ3.multi_obj import multi_objective_reward


def get_environments(type='normal', shuffled=False):
    logging.info("Loading market data...")
    forex_candle_data = ForexCandleData.load(
        source="dukascopy",
        instrument="EURUSD",
        granularity=Timeframe.M15,
        start_time=datetime(2017, 1, 1, 22, 0, 0, 0),
        end_time=datetime(2024, 12, 31, 21, 45, 0, 0),
    )

    logging.info("Setting up feature engineer...")
    market_feature_engineer = get_feature_engineer()

    logging.info("Setting up stepwise feature engineer...")
    agent_feature_engineer = StepwiseFeatureEngineer()
    agent_feature_engineer.add(["current_exposure"], calculate_current_exposure)

    logging.info("Creating environments...")

    train_env, eval_env = ForexEnv.create_train_eval_envs(
        split_ratio=0.7,
        forex_candle_data=forex_candle_data,
        market_feature_engineer=market_feature_engineer,
        agent_feature_engineer=agent_feature_engineer,
        initial_capital=10_000.0,
        transaction_cost_pct=0.0,
        n_actions=3,  # [-1, 1]
        custom_reward_function=equity_change,
        shuffled=shuffled,
    )

    if type == 'normal':
        logging.info("Environments created.")
        return train_env, eval_env

    else:

        reward_net_train = RewardNet(
            obs_dim=train_env.observation_space.shape[0],
            n_actions=3,
            model_type="mlp"  # "timesnet" | "wftnet" | "nlinear"
        )

        reward_net_eval = RewardNet(
            obs_dim=eval_env.observation_space.shape[0],
            n_actions=3,
            model_type="mlp"  # "timesnet" | "wftnet" | "nlinear"
        )

        train_self_env = SelfRewardingEnv(
            train_env,
            reward_net_train,
            lambda e: expert_reward_vector(e, label="minmax")
        )
        eval_self_env = SelfRewardingEnv(
            eval_env,
            reward_net_eval,
            lambda e: expert_reward_vector(e, label="minmax")
        )
        logging.info("Environments created.")
        return train_self_env, eval_self_env


def get_feature_engineer():
    """
    Returns a FeatureEngineer that constructs exactly the four groups of features used
    in Zhang et al. (2019):

      1) Price Momentum (pct‐changes)
      2) Trend / Moving Averages (EMA ratios, BB width, MACD, ADX)
      3) Momentum/Oscillators (RSI, Stochastic‐K/D, CCI)
      4) Volatility (ATR‐ratio)

    Usage:
        market_df = your_raw_OHLCV_dataframe
        fe = get_feature_engineer()
        feature_df = fe.run(market_df, remove_original_columns=True)
    """
    fe = FeatureEngineer()

    # 1) Price Momentum (Pct Change)
    def _feat_pct_change(df):
        # 1‐bar pct change
        copy_column(df, "close_bid", "close_pct_change_1")
        as_pct_change(df, "close_pct_change_1", periods=1)

        # 5‐bar pct change
        copy_column(df, "close_bid", "close_pct_change_5")
        as_pct_change(df, "close_pct_change_5", periods=5)

        # 14‐bar historic pct change (multiplied by 100 inside function)
        historic_pct_change(df, window=14)

        # Normalize via rolling min‐max over last 500 bars
        for col in ["close_pct_change_1", "close_pct_change_5", "historic_pct_change_14"]:
            as_min_max_window(df, column=col, window=500)

    fe.add(_feat_pct_change)

    # 2) Trend / Moving Averages
    def _feat_trend(df):
        # EMA 20 & EMA 50, then price/EMA ratios
        ema(df, window=20)
        as_ratio_of_other_column(df, "ema_20_close_bid", "close_bid")
        ema(df, window=50)
        as_ratio_of_other_column(df, "ema_50_close_bid", "close_bid")

        # Bollinger Bands width (normalized)
        bollinger_bands(df, window=20, num_std_dev=2.0)
        # width = (upper − lower) / middle
        df["bb_width_20"] = (df["bb_upper_20"] - df["bb_lower_20"]) / df["sma_20_close_bid"]
        df["bb_width_20"] = df["bb_width_20"].fillna(0.0)
        as_z_score(df, "bb_width_20", window=500)

        # MACD histogram, then z‐score normalize
        macd(df, short_window=12, long_window=26, signal_window=9)
        as_z_score(df, "macd_hist", window=500)

        # ADX (trend strength), z‐score normalize
        adx(df, window=14)
        as_z_score(df, "adx", window=500)

    fe.add(_feat_trend)

    # 3) Momentum / Oscillators
    def _feat_oscillators(df):
        # RSI(14) normalized to [0,1]
        rsi(df, window=14)
        df["rsi_14"] = df["rsi_14"].fillna(50.0)
        as_min_max_fixed(df, "rsi_14", 0, 100)

        # Stochastic %K and %D (14), normalize [0,100]
        stochastic_oscillator(df, window=14)
        df["stoch_k"] = df["stoch_k"].fillna(50.0)
        df["stoch_d"] = df["stoch_d"].fillna(50.0)
        as_min_max_fixed(df, "stoch_k", 0, 100)
        as_min_max_fixed(df, "stoch_d", 0, 100)

        # CCI(20), z‐score normalize
        cci(df, window=20)
        as_z_score(df, "cci_20", window=500)

    fe.add(_feat_oscillators)

    # 4) Volatility (ATR / Price)
    def _feat_volatility(df):
        atr(df, window=14)  # adds “atr_14”
        df["atr_ratio_14"] = df["atr_14"] / df["close_bid"]
        df["atr_ratio_14"] = df["atr_ratio_14"].fillna(0.0)
        as_z_score(df, "atr_ratio_14", window=500)

    fe.add(_feat_volatility)

    return fe


def get_model(mod, env: ForexEnv):
    logging.info("Creating model...")

    if mod == 'A2C':
        model = A2C(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-4,
            n_steps=128,
            gamma=0.3,
            vf_coef=0.1,
            ent_coef=0.0,
            verbose=1,
        )

    else:
        policy_kwargs = dict(net_arch=[12, 8], optimizer_class=optim.Adam, activation_fn=LeakyReLU)
        model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=0.0001,
            buffer_size=5000,
            learning_starts=480,
            batch_size=32,
            tau=1.0,
            gamma=0.9,
            train_freq=32,
            target_update_interval=500,
            exploration_fraction=0.5,
            exploration_initial_eps=1.0,
            exploration_final_eps=0.05,
            policy_kwargs=policy_kwargs,
            verbose=0,
            seed=42,
        )

    logging.info("Model created.")

    return model
