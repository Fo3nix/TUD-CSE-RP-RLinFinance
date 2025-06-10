import numpy as np
from common.envs.forex_env import ForexEnv, AgentDataCol


def equity_change(env: ForexEnv) -> float:
    """
    Calculate the change in equity from the start to the end of the episode.
    """
    current_time_step = env.current_step
    current_close_equity = env.agent_data[current_time_step, AgentDataCol.equity_close]
    previous_close_equity = env.agent_data[current_time_step - 1, AgentDataCol.equity_close]
    return current_close_equity - previous_close_equity


def log_equity_change(env: ForexEnv) -> float:
    """
    Calculate the log change in equity from the start to the end of the episode.
    """
    current_time_step = env.current_step
    current_close_equity = env.agent_data[current_time_step, AgentDataCol.equity_close]
    previous_close_equity = env.agent_data[current_time_step - 1, AgentDataCol.equity_close]

    if previous_close_equity <= 0:
        return 0.0  # Avoid log(0) or negative values

    return (current_close_equity / previous_close_equity) - 1.0

# # ------------------------------------------------------------------ #
# # 1) Profit-based rewards
# # ------------------------------------------------------------------ #
# def profit(env: ForexEnv) -> float:
#     """Absolute Δ-equity since the previous step."""
#     t = env.current_step
#     if t == 0:
#         return 0.0
#     return float(
#         env.agent_data[t, AgentDataCol.equity_close] -
#         env.agent_data[t - 1, AgentDataCol.equity_close]
#     )
#
#
# def log_profit(env: ForexEnv) -> float:
#     """Log-return style reward  ln(Eₜ / Eₜ₋₁)."""
#     t = env.current_step
#     if t == 0:
#         return 0.0
#     prev = env.agent_data[t - 1, AgentDataCol.equity_close]
#     return 0.0 if prev <= 0 else float(
#         np.log(env.agent_data[t, AgentDataCol.equity_close] / prev)
#     )
