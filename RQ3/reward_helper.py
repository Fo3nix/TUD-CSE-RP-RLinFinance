import numpy as np
from common.envs.forex_env import ForexEnv, AgentDataCol


def equity_window(env: ForexEnv, window: int | None):
    """
    Slice of equity_close from (current_step-window) … current_step  (inclusive).
    If window is None/0, returns the whole history available so far.
    """
    cur = env.agent_data[env.n_steps, AgentDataCol.pre_action_equity]
    start = max(0, cur - window) if window and window > 0 else 0
    return env.agent_data[start:cur + 1, AgentDataCol.equity_close]