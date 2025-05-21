from .profit_reward import profit_reward
from .risk_reward import sharpe_reward_dense
from .sparse_reward import sparse_wrapper
from .multi_objective import multi_objective_reward
from .imitation_reward import imitation_reward

REGISTRY = {
    "profit": profit_reward,  # raw PnL
    "sharpe": sharpe_reward_dense,  # risk‑adjusted dense
    "sparse": sparse_wrapper,  # k‑step sparse
    "multi": multi_objective_reward,  # PnL + risk blend
    "imitation": imitation_reward,  # placeholder for GAIL‑style
    # "drawdown" : drawdown_penalty,         # example of a future addition
}


# ─────────────────────────── public accessor ──────────────────────────────────
def get_reward(name: str):
    try:
        return REGISTRY[name]
    except KeyError as exc:
        raise ValueError(
            f"[rewards] Unknown reward '{name}'. "
            f"Available: {list(REGISTRY.keys())}"
        ) from exc
