import numpy as np
from .base_env import MarketMakingBaseEnv


class OUVanillaEnv(MarketMakingBaseEnv):
    """
    Ornstein–Uhlenbeck (OU) mid-price process with constant volatility.

    Dynamics:
        dS = kappa * (mu - S) * dt + sigma * dW
    """

    def __init__(self, sigma=2.0, kappa=1.0, mu=100.0, **kwargs):
        """
        Parameters
        ----------
        sigma : float
            Volatility scale of the OU process.
        kappa : float
            Mean-reversion speed.
        mu : float
            Long-run mean level of the process.
        kwargs : dict
            Forwarded to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kappa = kappa
        self.mu = mu

    def _update_price(self):
        """
        One OU step:
            S <- S + kappa*(mu - S)*dt + sigma*sqrt(dt)*N(0,1)
        """
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.S_prev = self.S
        self.S = self.S + self.kappa * (self.mu - self.S) * self.dt + self.sigma * dW
