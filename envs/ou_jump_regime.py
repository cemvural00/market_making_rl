import numpy as np
from .base_env import MarketMakingBaseEnv


class OUJumpRegimeEnv(MarketMakingBaseEnv):
    """
    Ornstein–Uhlenbeck (OU) mid-price with:
      - Discrete volatility regimes (2-state Markov)
      - Jump diffusion

    Dynamics:
        dS = kappa*(mu - S)*dt + sigma_regime*dW + J

    where:
        sigma_regime = sigma_low (regime=0) or sigma_high (regime=1)
        J ~ Normal(jump_mean, jump_std) with prob jump_intensity*dt, else 0
    """

    def __init__(
        self,
        sigma_low=1.0,
        sigma_high=4.0,
        kappa=1.0,
        mu=100.0,
        jump_intensity=0.1,
        jump_mean=0.0,
        jump_std=5.0,
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
        jump_intensity : float
            Poisson jump intensity λ_jump.
        jump_mean : float
            Mean jump size.
        jump_std : float
            Std dev of jump size.
        transition_matrix : np.ndarray (2x2), optional
            Regime transition matrix.
            Defaults to:
                [[0.95, 0.05],
                 [0.10, 0.90]]
        initial_regime : int
            Starting regime (0 or 1).
        kwargs : dict
            Forwarded to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)

        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.kappa = kappa
        self.mu = mu

        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

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

    def _draw_jump(self):
        """Sample a single jump size."""
        return self.rng.normal(self.jump_mean, self.jump_std)

    def _update_price(self):
        """
        One OU + regime vol + jump step:

            1. Update regime
            2. Choose sigma = sigma_low or sigma_high
            3. OU diff: kappa*(mu - S)*dt + sigma*sqrt(dt)*N(0,1)
            4. Jump: J with prob λ_jump*dt, else 0
            5. S <- S + diff + jump
        """
        # 1) Regime update
        self._update_regime()

        # 2) Regime-dependent volatility
        sigma = self.sigma_low if self.regime == 0 else self.sigma_high

        # 3) OU diffusion part
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        dS_diff = self.kappa * (self.mu - self.S) * self.dt + sigma * dW

        # 4) Jump part
        if self.rng.uniform() < self.jump_intensity * self.dt:
            J = self._draw_jump()
        else:
            J = 0.0

        # 5) Apply update
        self.S_prev = self.S
        self.S = self.S + dS_diff + J

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.regime = self.initial_regime
        return obs, info
