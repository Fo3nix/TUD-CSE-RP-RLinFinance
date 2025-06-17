import numpy as np

from common.constants import MarketDataCol
from common.envs.forex_env import ForexEnv, AgentDataCol
from RQ3.reward_helper import equity_window
from common.envs.rewards import equity_change


def risk_adjusted_return(env: ForexEnv) -> float:
    """
    Calculate the risk-adjusted return based on the Sharpe ratio.
    """
    current_time_step = env.n_steps

    current_equity_change = equity_change(env)
    volatility = env.agent_data[current_time_step, AgentDataCol.equity_high] - env.agent_data[
        current_time_step, AgentDataCol.equity_low]
    epsilon = 1e-5  # Small value to avoid division by zero

    return current_equity_change / (volatility + epsilon)


def volatility_scaled_reward(env,
                             vol_target: float = 0.10,
                             bp: float = 0.0020,
                             window: int = 60,
                             lam: float = 0.94) -> float:
    """
    Custom reward function that replicates Equation (4) in Zhang et al. (2019).

    Reward formula (for t >= window and t >= 2):
      R_t = (σ_tgt / σ_{t-1}) * [ A_{t-1} * (p_t - p_{t-1})  -  bp * p_{t-1} * |A_{t-1} - A_{t-2}| ]

    Parameters:
    - env: instance of ForexEnv, already stepped so that current_step = t.
    - vol_target: target volatility σ_tgt (e.g. 0.10 for 10%).
    - bp: transaction cost rate (e.g. 0.0020 for 0.20%).
    - window: lookback window (in bars) for ex‐ante EWMA volatility (e.g. 60).
    - lam: EWMA decay factor (e.g. 0.94).

    Returns:
    - A single float reward for time t. If t < max(window, 2), returns 0.0.
    """

    t = env.n_steps

    # Need at least window bars to estimate σ_{t-1}, and at least 2 steps to have A_{t-2}.
    if t < window or t < 2:
        return 0.0

    # --- 1) Extract A_{t-1} and A_{t-2} from agent_data ---
    # Note: agent_data was just written at index t, so
    #   agent_data[t-1, AgentDataCol.action] is A_{t-1],
    #   agent_data[t-2, AgentDataCol.action] is A_{t-2].
    A_tm1 = float(env.agent_data[t - 1, AgentDataCol.action])
    A_tm2 = float(env.agent_data[t - 2, AgentDataCol.action])

    # --- 2) Extract p_t and p_{t-1} from market_data (use close_bid) ---
    close_bid_t = float(env.market_data[t, MarketDataCol.close_bid])
    close_bid_tm1 = float(env.market_data[t - 1, MarketDataCol.close_bid])
    r_t = close_bid_t - close_bid_tm1

    # --- 3) Compute ex‐ante EWMA volatility σ_{t-1} using last `window` close_bid bars ---
    # We want returns over [t-window, …, t-1], so prices[i+1] - prices[i] for i = t-window .. t-2
    start_idx = t - window
    end_idx = t  # exclusive, so market_data[start_idx : end_idx] has length = window
    recent_prices = env.market_data[start_idx:end_idx, MarketDataCol.close_bid]  # shape = (window,)

    # Compute simple arithmetic returns of length (window-1)
    recent_returns = recent_prices[1:] - recent_prices[:-1]  # shape = (window-1,)

    # Build EWMA weights of length (window-1): w_i ∝ (1-lam) * lam^(i), with i = 0 for most recent.
    # We want weights such that the most recent return (i = window-2) gets the largest lam^0.
    # So build it reversed: index 0 corresponds to t-2 (most recent), index window-2 corresponds to t-window.
    L = len(recent_returns)  # = window - 1
    raw_weights = np.array([(1 - lam) * (lam ** i) for i in range(L)])
    # raw_weights[0] = (1-lam)*lam^0 corresponds to oldest return, so reverse:
    ewma_weights = raw_weights[::-1]
    ewma_weights = ewma_weights / ewma_weights.sum()

    # σ_{t-1} is the sqrt of weighted average of squared returns:
    sigma_tm1 = float(np.sqrt(np.dot(ewma_weights, recent_returns ** 2)))
    if sigma_tm1 < 1e-12:
        # If volatility is essentially zero (flat market), we give zero reward to avoid blow‐ups.
        return 0.0

    # --- 4) Compute volatility‐scaling factor ---
    vol_scaling = vol_target / sigma_tm1

    # --- 5) Compute transaction cost term: bp * p_{t-1} * |A_{t-1} - A_{t-2}| ---
    trans_cost = bp * close_bid_tm1 * abs(A_tm1 - A_tm2)

    # --- 6) Compute final reward ---
    # R_t = vol_scaling * [ A_{t-1} * r_t  -  trans_cost ]
    reward = vol_scaling * (A_tm1 * r_t - trans_cost)
    return float(reward)

def hybrid_reward(env):
    sr = sharpe_ratio_reward(env)
    immediate = env.agent_data[env.n_steps + 1, AgentDataCol.equity_close] - \
                env.agent_data[env.n_steps, AgentDataCol.equity_close]
    return sr + 0.1 * immediate

def sharpe_ratio_reward(
    env,
    window: int = 100,
    risk_free_rate: float = 0.0,
    annual_factor: int = 252
) -> float:
    """
    Rolling Sharpe ratio of equity_close returns.
    Returns 0 if not enough data yet or σ == 0.
    """
    cur = int(env.n_steps)
    if cur < 2:
        return 0.0

    start = max(0, cur - window + 1)
    series = env.agent_data[start:cur + 1, AgentDataCol.equity_close]

    if len(series) < 2:
        return 0.0

    # use recent returns only (drop burn-in)
    rets = np.diff(series) / series[:-1]
    if len(rets) < 5:
        return 0.0

    excess = rets - risk_free_rate
    sigma = np.std(excess, ddof=1)
    if sigma == 0:
        return 0.0

    sharpe = np.sqrt(annual_factor) * np.mean(excess) / sigma
    return float(sharpe)

# ------------------------------------------------------------------ #
# 3) Mean-variance utility reward
# ------------------------------------------------------------------ #
def mean_variance_reward(
    env,
    window: int = 30,
    lam: float = 1.0
) -> float:
    """
    μ − λ·σ² over a rolling window of equity_close returns.
    Returns 0.0 if insufficient data.
    """
    cur = int(env.n_steps)
    if cur < 2:
        return 0.0

    start = max(0, cur - window + 1)
    series = env.agent_data[start:cur + 1, AgentDataCol.equity_close]

    if len(series) < 2:
        return 0.0

    rets = np.diff(series) / series[:-1]
    if len(rets) < 2:   # 🔒 this is the key fix
        return 0.0

    mu = np.mean(rets)
    var = np.var(rets, ddof=1)
    return float(mu - lam * var)



# ------------------------------------------------------------------ #
# 4) CVaR-based reward
# ------------------------------------------------------------------ #
def cvar_reward(
        env: ForexEnv,
        window: int = 60,
        alpha: float = 0.05  # tail-probability level
) -> float:
    """
    Negative Conditional Value-at-Risk (CVaR) of returns.
    Reward is –CVaR, so *smaller* expected tail losses ⇒ *larger* reward.
    """
    cur = int(env.n_steps)
    if cur < 2:
        return 0.0
    start = max(0, cur - window + 1)
    series = env.agent_data[start:cur + 1, AgentDataCol.equity_close]
    if len(series) < 2:
        return 0.0

    rets = np.diff(series) / series[:-1]
    if len(rets) < 2:   # 🔒 this is the key fix
        return 0.0
    var_level = np.quantile(rets, alpha)
    tail = rets[rets <= var_level]
    if tail.size == 0:
        return 0.0
    cvar = tail.mean()  # expected loss in the worst α fraction
    return float(-cvar)
