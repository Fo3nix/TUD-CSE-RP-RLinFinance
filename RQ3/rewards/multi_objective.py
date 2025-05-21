
import numpy as np
def multi_objective_reward(env, raw_pnl, alpha=0.5):
    # alpha weight for risk-adjusted term
    env.history.append(raw_pnl)
    if len(env.history) < 50:
        risk_term = 0.0
    else:
        returns = np.array(env.history[-50:])
        risk_term = returns.mean() / (returns.std() + 1e-8)
    return alpha * raw_pnl + (1 - alpha) * risk_term
