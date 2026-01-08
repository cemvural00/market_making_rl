import numpy as np

def compute_basic_metrics(pnls):
    mean = pnls.mean()
    std = pnls.std()
    sharpe = mean / (std + 1e-12)

    # VaR (1-day)
    var_95 = np.quantile(pnls, 0.05)
    var_99 = np.quantile(pnls, 0.01)

    # Expected shortfall
    es_95 = pnls[pnls <= var_95].mean()
    es_99 = pnls[pnls <= var_99].mean()

    return {
        "mean": mean,
        "std": std,
        "sharpe": sharpe,
        "var_95": var_95,
        "var_99": var_99,
        "es_95": es_95,
        "es_99": es_99
    }
