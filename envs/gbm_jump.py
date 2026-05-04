import numpy as np
from .base_env import MarketMakingBaseEnv


class GBMJumpEnv(MarketMakingBaseEnv):
    """
    Geometric Brownian Motion with jump diffusion (Merton model).

    Dynamics:
        S <- S * exp( (mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*N(0,1) + J )

    where J is the log-jump size (J ~ N(jump_mean, jump_std^2) when jump occurs).
    Jumps are multiplicative (log-normal), ensuring S > 0 always.
    """

    _normalize_lags_by_current_price = True

    def __init__(
        self,
        sigma=0.02,
        mu=0.0,
        jump_intensity=0.1,
        jump_mean=0.0,
        jump_std=0.05,  # Changed default: log-jump std (percentage scale)
        **kwargs
    ):
        """
        Parameters
        ----------
        sigma : float
            Volatility (percentage).
        mu : float
            Drift (percentage).
        jump_intensity : float
            Poisson jump intensity λ (expected jumps per unit time).
        jump_mean : float
            Mean of log-jump size distribution.
            For percentage jumps, typically small (e.g., 0.0 for symmetric).
        jump_std : float
            Std dev of log-jump size distribution.
            For percentage jumps, typically 0.05 (5% log-volatility).
        kwargs : dict
            Forwarded to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)

        self.sigma = sigma
        self.mu = mu

        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def _draw_jump(self):
        """
        Sample log-jump size J ~ N(jump_mean, jump_std^2).
        This will be used multiplicatively: S <- S * exp(J)
        """
        return self.rng.normal(self.jump_mean, self.jump_std)

    def _update_price(self):
        """
        GBM with multiplicative (log-normal) jumps:
            S <- S * exp((μ - 0.5σ²)dt + σ·dW + J)
        where J is the log-jump size (0 if no jump, or N(jump_mean, jump_std²) if jump occurs).
        """
        dt = self.dt
        sigma = self.sigma
        mu = self.mu

        dW = self.rng.normal(0.0, np.sqrt(dt))
        self.S_prev = self.S

        # GBM diffusion term
        gbm_diffusion = (mu - 0.5 * sigma**2) * dt + sigma * dW

        # Jump term (log-jump size)
        if self.rng.uniform() < self.jump_intensity * dt:
            J_log = self._draw_jump()  # Log-jump size
        else:
            J_log = 0.0

        # Multiplicative update: S <- S * exp(diffusion + jump)
        self.S = self.S * np.exp(gbm_diffusion + J_log)
