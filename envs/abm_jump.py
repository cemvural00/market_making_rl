import numpy as np
from .base_env import MarketMakingBaseEnv


class ABMJumpEnv(MarketMakingBaseEnv):
    """
    Arithmetic Brownian Motion (ABM) with Jump Diffusion.

    Mid-price evolves as:
        dS = mu*dt + sigma*dW + J*dN

    where:
        dN ~ Bernoulli(jump_intensity * dt)
        J ~ Normal(jump_mean, jump_std)
    """

    def __init__(
        self,
        sigma=2.0,
        mu=0.0,
        jump_intensity=0.1,   # Expected jumps per unit time
        jump_mean=0.0,
        jump_std=5.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        sigma : float
            Diffusive volatility.
        mu : float
            Drift term (default 0).
        jump_intensity : float
            Poisson intensity λ for jumps.
        jump_mean : float
            Mean of jump size distribution.
        jump_std : float
            Std dev of jump size distribution.
        kwargs : dict
            Passed to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)

        self.sigma = sigma
        self.mu = mu

        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def _draw_jump(self):
        """Sample a single jump size from N(jump_mean, jump_std)."""
        return self.rng.normal(self.jump_mean, self.jump_std)

    def _update_price(self):
        """
        One-step update:

            S ← S + mu*dt + sigma*sqrt(dt)*N(0,1) + J

        where J = 0 with probability (1 - jump_intensity*dt),
              or a normally distributed jump otherwise.
        """
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.S_prev = self.S

        # Diffusion term
        dS_diff = self.mu * self.dt + self.sigma * dW

        # Jump term
        if self.rng.uniform() < self.jump_intensity * self.dt:
            J = self._draw_jump()
        else:
            J = 0.0

        self.S = self.S + dS_diff + J
