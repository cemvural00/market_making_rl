import numpy as np
from .base_env import MarketMakingBaseEnv


class ABMVanillaEnv(MarketMakingBaseEnv):
    """
    Arithmetic Brownian Motion (ABM) environment with optional drift.
    Default is drift-free: dS = sigma * dW
    """

    def __init__(self, sigma=2.0, mu=0.0, **kwargs):
        super().__init__(**kwargs)
        self.sigma = sigma
        self.mu = mu

    def _update_price(self):
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.S_prev = self.S
        self.S = self.S + self.mu * self.dt + self.sigma * dW
