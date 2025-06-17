# oracle_labels.py
import numpy as np

def oracle_labels(prices, commission_bps: float, final_state: int = 0):
    """
    Dynamic-programming implementation of Algorithm 1:contentReference[oaicite:0]{index=0}.
    prices : 1-D array of close prices (len T)
    commission_bps : commission in basis-points Θ = ϑ
    final_state : y_T  (0 = flat, 1 = long) – choose 0 for intraday flat-close
    Returns
    -------
    y : (T,) array of {0,1} labels
    """
    bps = commission_bps / 10_000
    T = len(prices)
    S = np.full((2, T), -np.inf)           # state-value matrix
    S[final_state, -1] = 0.0               # terminal reward = 0

    # pre-compute transition gains P_{i,j}^{t-1}  (Eq 2-4):contentReference[oaicite:1]{index=1}
    P = np.zeros((2, 2, T-1))
    for t in range(T-1):
        ret = (prices[t+1] / prices[t]) - 1
        P[0, 1, t] = 0                     # opening a long today starts counting tomorrow
        P[1, 0, t] = ((1 - bps) * (1 + ret) - 1)    # close long & pay commission
        P[1, 1, t] = ret                            # in-position
        P[0, 0, t] = 0                              # flat

    # backward DP
    for t in range(T-2, -1, -1):
        for j in (0, 1):
            S[j, t] = max(S[i, t+1] + P[i, j, t] for i in (0, 1))

    # reconstruct labels
    y = np.empty(T, dtype=int)
    y[-1] = final_state
    for t in range(T-2, -1, -1):
        j = y[t+1]
        y[t] = max((i for i in (0, 1)),
                   key=lambda i: S[i, t+1] + P[i, j, t])

    return y
