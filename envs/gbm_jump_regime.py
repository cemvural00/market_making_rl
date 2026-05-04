import numpy as np
from .base_env import MarketMakingBaseEnv


class GBMJumpRegimeEnv(MarketMakingBaseEnv):
    """
    Geometric Brownian Motion with:
      - Discrete volatility regimes (2-state Markov chain)
      - Multiplicative jump diffusion (Merton model)

    Dynamics:
        S <- S * exp( (mu - 0.5*sigma_regime^2)*dt + sigma_regime*sqrt(dt)*N(0,1) + J )

    where:
        - sigma_regime depends on current regime (sigma_low or sigma_high)
        - J is log-jump size (J ~ N(jump_mean, jump_std^2) when jump occurs)
        - Jumps are multiplicative, ensuring S > 0 always
    """

    _normalize_lags_by_current_price = True

    def __init__(
        self,
        sigma_low=0.01,
        sigma_high=0.05,
        mu=0.0,
        jump_intensity=0.1,
        jump_mean=0.0,
        jump_std=0.05,  # Changed default: log-jump std (percentage scale)
        transition_matrix=None,
        initial_regime=0,
        **kwargs
    ):
        """
        Parameters
        ----------
        sigma_low : float
            Volatility in regime 0 (percentage).
        sigma_high : float
            Volatility in regime 1 (percentage).
        mu : float
            Drift (percentage).
        jump_intensity : float
            Poisson jump intensity λ (expected jumps per unit time).
        jump_mean : float
            Mean of log-jump size distribution.
        jump_std : float
            Std dev of log-jump size distribution (typically 0.05 for 5% log-volatility).
        transition_matrix : np.ndarray (2x2), optional
            Regime transition matrix. Defaults to [[0.95, 0.05], [0.10, 0.90]].
        initial_regime : int
            Starting regime (0 or 1).
        kwargs : dict
            Forwarded to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)

        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.mu = mu

        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        if transition_matrix is None:
            self.transition_matrix = np.array([[0.95, 0.05],
                                               [0.10, 0.90]])
        else:
            self.transition_matrix = np.array(transition_matrix)

        self.initial_regime = initial_regime
        self.regime = initial_regime

    def _update_regime(self):
        """Evolve volatility regime via 2-state Markov chain."""
        probs = self.transition_matrix[self.regime]
        self.regime = self.rng.choice([0, 1], p=probs)

    def _draw_jump(self):
        """
        Sample log-jump size J ~ N(jump_mean, jump_std^2).
        This will be used multiplicatively: S <- S * exp(J)
        """
        return self.rng.normal(self.jump_mean, self.jump_std)

    def _update_price(self):
        """
        GBM with regime-switching volatility and multiplicative jumps:
            1. Update regime
            2. S <- S * exp((μ - 0.5σ²)dt + σ·dW + J)
        where J is log-jump size (0 if no jump, or N(jump_mean, jump_std²) if jump occurs).
        """
        dt = self.dt

        # 1. Update regime
        self._update_regime()
        sigma = self.sigma_low if self.regime == 0 else self.sigma_high

        # 2. GBM diffusion term
        dW = self.rng.normal(0.0, np.sqrt(dt))
        gbm_diffusion = (self.mu - 0.5 * sigma**2) * dt + sigma * dW

        # 3. Jump term (log-jump size)
        if self.rng.uniform() < self.jump_intensity * dt:
            J_log = self._draw_jump()  # Log-jump size
        else:
            J_log = 0.0

        # Multiplicative update: S <- S * exp(diffusion + jump)
        self.S_prev = self.S
        self.S = self.S * np.exp(gbm_diffusion + J_log)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self.regime = self.initial_regime
        return obs, info
