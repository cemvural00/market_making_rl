import numpy as np
from .base_env import MarketMakingBaseEnv


class ABMRegimeEnv(MarketMakingBaseEnv):
    """
    Arithmetic Brownian Motion (ABM) with discrete volatility regimes.

    Price evolves as:
        dS = mu*dt + sigma_regime * sqrt(dt) * N(0,1)

    Regime evolves as a 2-state Markov chain:
        regime ∈ {0, 1}

    Transition matrix:
        [ [p00, p01],
          [p10, p11] ]

    where p01 = P(0→1), p10 = P(1→0)
    """

    def __init__(
        self,
        sigma_low=1.0,
        sigma_high=4.0,
        mu=0.0,
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
        mu : float
            Drift term (default 0).
        transition_matrix : np.array shape (2,2)
            Transition probabilities between regimes.
            If None, defaults to:
                [[0.95, 0.05],
                 [0.10, 0.90]]
        initial_regime : int (0 or 1)
            Starting regime for each episode.
        kwargs : dict
            Passed to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)

        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.mu = mu

        # If no matrix provided, use a reasonable default
        if transition_matrix is None:
            self.transition_matrix = np.array([[0.95, 0.05],
                                               [0.10, 0.90]])
        else:
            self.transition_matrix = np.array(transition_matrix)

        assert self.transition_matrix.shape == (2, 2), \
            "transition_matrix must be 2x2"

        self.initial_regime = initial_regime
        self.regime = initial_regime

    # ----------------------------------------------------
    #   Regime update function
    # ----------------------------------------------------
    def _update_regime(self):
        """
        Evolve regime using a 2-state Markov chain.

        regime ∈ {0,1}

        next_regime ~ Categorical(transition_matrix[regime])
        """
        probs = self.transition_matrix[self.regime]
        self.regime = self.rng.choice([0, 1], p=probs)

    # ----------------------------------------------------
    #   Price update
    # ----------------------------------------------------
    def _update_price(self):
        """
        Price evolves via ABM:

            dS = mu*dt + sigma_regime*sqrt(dt)*N(0,1)

        where sigma depends on the current volatility regime.
        """
        # 1. Update volatility regime
        self._update_regime()

        # 2. Select volatility
        sigma = self.sigma_low if self.regime == 0 else self.sigma_high

        # 3. Evolve price
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.S_prev = self.S
        self.S = self.S + self.mu * self.dt + sigma * dW

    # ----------------------------------------------------
    #   Reset: we override to restore the initial regime
    # ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed)
        self.regime = self.initial_regime
        return obs, info
