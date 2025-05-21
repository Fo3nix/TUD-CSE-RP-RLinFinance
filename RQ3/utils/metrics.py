import numpy as np


def _annualize(value, periods):
    return value * np.sqrt(periods)


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    excess = returns - risk_free_rate / periods_per_year
    return _annualize(excess.mean() / returns.std(ddof=1), periods_per_year)


def sortino_ratio(returns, target=0.0, periods_per_year=252):
    downside = returns[returns < target]
    denom = np.sqrt((downside ** 2).sum() / len(returns))
    return _annualize((returns.mean() - target) / denom, periods_per_year)


def cvar(returns, alpha=0.05):
    threshold = np.quantile(returns, alpha)
    return returns[returns <= threshold].mean()


def max_drawdown(cumulative):
    running_max = np.maximum.accumulate(cumulative)
    drawdown = 1 - cumulative / running_max
    return drawdown.max()
