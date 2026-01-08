import numpy as np
from .base_env import MarketMakingBaseEnv


class OURegimeEnv(MarketMakingBaseEnv):
    """
    Ornstein–Uhlenbeck (OU) mid-price with discrete volatility regimes.

    Dynamics:
        dS = kappa*(mu - S)*dt + sigma_regime * dW

    where sigma_regime = sigma_low (regime=0) or sigma_high (regime=1),
    and regime follows a 2-state Markov chain.
    """

    def __init__(
        self,
        sigma_low=1.0,
        sigma_high=4.0,
        kappa=1.0,
        mu=100.0,
        transition_matrix=None,
        initial_regime=0,
        **kwargs
    ):
        """
        Parameters
        ----------
        sigma_low : float
            Volatility in regime 0.
        sigma_high : float
            Volatility in regime 1.
        kappa : float
            Mean-reversion speed.
        mu : float
            Long-run mean.
        transition_matrix : np.ndarray (2x2), optional
            Markov regime transition matrix.
            If None, defaults to:
                [[0.95, 0.05],
                 [0.10, 0.90]]
        initial_regime : int
            Starting regime at reset (0 or 1).
        kwargs : dict
            Forwarded to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)

        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.kappa = kappa
        self.mu = mu

        if transition_matrix is None:
            self.transition_matrix = np.array([[0.95, 0.05],
                                               [0.10, 0.90]])
        else:
            self.transition_matrix = np.array(transition_matrix)

        assert self.transition_matrix.shape == (2, 2), \
            "transition_matrix must be 2x2"

        self.initial_regime = initial_regime
        self.regime = initial_regime

    def _update_regime(self):
        """Evolve volatility regime via 2-state Markov chain."""
        probs = self.transition_matrix[self.regime]
        self.regime = self.rng.choice([0, 1], p=probs)

    def _update_price(self):
        """
        OU step with regime-dependent volatility:

            1. Update regime
            2. sigma = sigma_low or sigma_high
            3. S <- S + kappa*(mu - S)*dt + sigma*sqrt(dt)*N(0,1)
        """
        self._update_regime()

        sigma = self.sigma_low if self.regime == 0 else self.sigma_high

        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.S_prev = self.S
        self.S = self.S + self.kappa * (self.mu - self.S) * self.dt + sigma * dW

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.regime = self.initial_regime
        return obs, info
