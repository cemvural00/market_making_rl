import numpy as np
from .base_env import MarketMakingBaseEnv


class OUJumpEnv(MarketMakingBaseEnv):
    """
    Ornstein–Uhlenbeck (OU) mid-price with jump diffusion.

    Dynamics:
        dS = kappa*(mu - S)*dt + sigma*dW + J

    where:
        with probability jump_intensity * dt, J ~ Normal(jump_mean, jump_std)
        otherwise J = 0.
    """

    def __init__(
        self,
        sigma=2.0,
        kappa=1.0,
        mu=100.0,
        jump_intensity=0.1,
        jump_mean=0.0,
        jump_std=5.0,
        **kwargs
    ):
        """
        Parameters
        ----------
        sigma : float
            Volatility scale of OU.
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
        kwargs : dict
            Forwarded to MarketMakingBaseEnv.
        """
        super().__init__(**kwargs)
        self.sigma = sigma
        self.kappa = kappa
        self.mu = mu

        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

    def _draw_jump(self):
        """Sample a single jump size."""
        return self.rng.normal(self.jump_mean, self.jump_std)

    def _update_price(self):
        """
        One-step OU + jump:

            diff = kappa*(mu - S)*dt + sigma*sqrt(dt)*N(0,1)
            jump = J with prob λ_jump*dt else 0
            S <- S + diff + jump
        """
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        self.S_prev = self.S

        # OU diffusion part
        dS_diff = self.kappa * (self.mu - self.S) * self.dt + self.sigma * dW

        # Jump part
        if self.rng.uniform() < self.jump_intensity * self.dt:
            J = self._draw_jump()
        else:
            J = 0.0

        self.S = self.S + dS_diff + J
