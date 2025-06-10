import torch
import numpy as np


def expert_reward_vector(
        env,
        label: str = "return",  # "return" | "sharpe" | "minmax"
        k: int = 5,  # look-ahead for sharpe
        m: int = 10,  # look-ahead for min-max
):
    """
    Build a 3-action reward vector [short, hold, long] in the order
    expected by the environment’s action space.

    • SHORT  (action 0) → -1 position
    • HOLD   (action 1) →  0 position
    • LONG   (action 2) → +1 position
    """
    pos = torch.tensor([-1.0, 0.0, 1.0])  # per-action multipliers
    t = env.current_step  # <- FIX: define t

    # ---- obtain close-price history ---------------------------------
    try:
        prices = env.market_data[:, env.price_col_close]
    except AttributeError:
        prices = env.prices  # adjust to your env layout

    # -----------------------------------------------------------------
    if label == "return":
        if t == 0:  # nothing to compare yet
            core = 0.0
        else:
            pct = (prices[t] - prices[t - 1]) / prices[t - 1]
            core = pct * 100  # paper scales *100
        r_vec = pos * core

    elif label == "sharpe":
        end = min(len(prices), t + k + 1)
        future = (prices[t + 1: end] - prices[t]) / prices[t]
        if future.size == 0:
            core = 0.0
        else:
            mu = future.mean()
            sigma = future.std() + 1e-12
            core = mu / sigma  # short-term Sharpe
        r_vec = pos * core

    elif label == "minmax":
        end = min(len(prices), t + m + 1)
        fut = (prices[t + 1: end] - prices[t]) / prices[t]
        if fut.size == 0:
            core = 0.0
        else:
            maxR, minR = fut.max(), fut.min()
            if (maxR > 0) or (maxR + minR > 0):
                core = maxR
            elif (minR < 0) or (maxR + minR < 0):
                core = minR
            else:
                core = -(maxR - minR)  # flat-market penalty
        r_vec = pos * core

    else:
        raise ValueError("label must be 'return', 'sharpe', or 'minmax'")

    return r_vec.float()  # tensor, shape (3,)
