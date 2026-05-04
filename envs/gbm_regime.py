import numpy as np
from .base_env import MarketMakingBaseEnv


class GBMRegimeEnv(MarketMakingBaseEnv):
    """
    Geometric Brownian Motion with regime-switching volatility.

    Dynamics:
        sigma_regime ∈ {sigma_low, sigma_high}
        S <- S * exp( (mu - 0.5*sigma_regime^2)*dt + sigma_regime*sqrt(dt)*N(0,1) )
    """

    _normalize_lags_by_current_price = True

    def __init__(
        self,
        sigma_low=0.01,
        sigma_high=0.05,
        mu=0.0,
        transition_matrix=None,
        initial_regime=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.mu = mu

        if transition_matrix is None:
            self.transition_matrix = np.array([[0.95, 0.05],
                                               [0.10, 0.90]])
        else:
            self.transition_matrix = np.array(transition_matrix)

        self.initial_regime = initial_regime
        self.regime = initial_regime

    def _update_regime(self):
        probs = self.transition_matrix[self.regime]
        self.regime = self.rng.choice([0, 1], p=probs)

    def _update_price(self):
        self._update_regime()

        sigma = self.sigma_low if self.regime == 0 else self.sigma_high
        mu = self.mu
        dt = self.dt

        dW = self.rng.normal(0.0, np.sqrt(dt))
        self.S_prev = self.S

        self.S = self.S * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.regime = self.initial_regime
        return obs, info
