import numpy as np
from .base_env import MarketMakingBaseEnv


class GBMVanillaEnv(MarketMakingBaseEnv):
    """
    Geometric Brownian Motion (GBM) mid-price process.

    Dynamics:
        dS = mu * S * dt + sigma * S * dW
        S <- S * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*N(0,1) )
    """

    def __init__(self, sigma=0.02, mu=0.0, **kwargs):
        """
        Parameters
        ----------
        sigma : float
            Volatility (percentage).
        mu : float
            Drift (percentage).
        kwargs : dict
            Forwarded to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.mu = mu

    def _update_price(self):
        dt = self.dt
        sigma = self.sigma
        mu = self.mu

        dW = self.rng.normal(0.0, np.sqrt(dt))

        # GBM exponential step
        self.S_prev = self.S
        self.S = self.S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
