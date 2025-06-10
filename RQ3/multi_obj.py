import numpy as np

from common.constants import MarketDataCol
from common.envs.forex_env import ForexEnv, AgentDataCol
from RQ3.profit_based import equity_change
from RQ3.reward_helper import equity_window


def multi_objective_reward(
        env: ForexEnv,
        w_profit: float = 1.0,
        w_risk: float = 1.0,
        w_tc: float = 1.0,
        w_dd: float = 1.0,
        risk_window: int = 30,
        dd_window: int | None = None,
) -> float:
    """
    Weighted combination of:
      • Profit  (Δ-equity)
      • Risk    (rolling variance of returns)
      • TxCost  (transaction/commission paid at current step)
      • Drawdown (current equity drawdown)

    Positive weights *penalise* risk, costs, and drawdown because those
    components are defined as non-positive (≤0).  Feel free to pass
    negative weights if you prefer an alternative sign convention.
    """
    # 1) Profit component (can be positive or negative)
    r_profit = equity_change(env)  # Δ-equity since last step

    # 2) Risk penalty = –variance of recent returns  (≤ 0)
    series = equity_window(env, risk_window)
    if len(series) > 1:
        rets = np.diff(series) / series[:-1]
        r_risk = -np.var(rets, ddof=1)
    else:
        r_risk = 0.0

    # 3) Transaction-cost penalty  (≤ 0)
    try:
        tc = env.agent_data[env.current_step, AgentDataCol.transaction_cost]
        r_tc = -float(tc)
    except (AttributeError, IndexError):
        # Fallback if the environment does not store per-step costs
        r_tc = 0.0

    # 4) Drawdown penalty = current (negative) drawdown  (≤ 0)
    eq = series[-1] if len(series) else env.agent_data[env.current_step, AgentDataCol.equity_close]
    max_eq = np.max(series) if len(series) else eq
    r_dd = (eq - max_eq) / max_eq if max_eq > 0 else 0.0  # 0 ≤ r_dd ≤ 0

    # Combine
    return (
            w_profit * r_profit +
            w_risk * r_risk +
            w_tc * r_tc +
            w_dd * r_dd
    )
