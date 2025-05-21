import numpy as np
from .global_history import GLOBAL_HISTORY

def sharpe_reward_dense(env, raw_pnl, windows=[10, 50, 100]):
    env.history.append(raw_pnl)

    sharpes = []
    for w in windows:
        data = env.history[-w:] if len(env.history) >= w else env.history
        returns = np.array(data)
        sharpe = returns.mean() / (returns.std() + 1e-8)
        sharpes.append(sharpe)

    # Scale Sharpe to ensure it influences learning
    scaled_sharpe = np.mean(sharpes) * 1000

    # Debug log
    if len(env.history) % 100 == 0:
        print(f"[DEBUG] Sharpe: {scaled_sharpe:.4f}, Raw PnL: {raw_pnl:.4f}")

    return scaled_sharpe

