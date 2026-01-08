import numpy as np
from .base_env import MarketMakingBaseEnv


class ABMJumpRegimeEnv(MarketMakingBaseEnv):
    """
    Arithmetic Brownian Motion (ABM) with:
      - Discrete volatility regimes (2-state Markov chain)
      - Jump diffusion component

    Price evolves as:
        dS = mu*dt + sigma_regime * sqrt(dt) * N(0,1) + J

    where:
        regime ∈ {0, 1} with Markov transitions
        sigma_regime = sigma_low (regime=0) or sigma_high (regime=1)
        J ~ Normal(jump_mean, jump_std) with probability jump_intensity * dt
    """

    def __init__(
        self,
        sigma_low=1.0,
        sigma_high=4.0,
        mu=0.0,
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
            Volatility in regime 0 (low-vol).
        sigma_high : float
            Volatility in regime 1 (high-vol).
        mu : float
            Drift term (default 0).
        jump_intensity : float
            Poisson jump intensity λ_jump.
            Effective per-step jump probability ≈ λ_jump * dt.
        jump_mean : float
            Mean of jump size distribution.
        jump_std : float
            Std dev of jump size distribution.
        transition_matrix : np.ndarray (2x2), optional
            Regime transition matrix:
                [[p00, p01],
                 [p10, p11]]
            If None, defaults to:
                [[0.95, 0.05],
                 [0.10, 0.90]]
        initial_regime : int (0 or 1)
            Starting regime at episode reset.
        kwargs : dict
            Forwarded to MarketMakingBaseEnv (S0, T, dt, A, k, base_delta, etc.).
        """
        super().__init__(**kwargs)

        # Volatility regimes
        self.sigma_low = sigma_low
        self.sigma_high = sigma_high
        self.mu = mu

        # Jumps
        self.jump_intensity = jump_intensity
        self.jump_mean = jump_mean
        self.jump_std = jump_std

        # Regime transition matrix
        if transition_matrix is None:
            self.transition_matrix = np.array([[0.95, 0.05],
                                               [0.10, 0.90]])
        else:
            self.transition_matrix = np.array(transition_matrix)

        assert self.transition_matrix.shape == (2, 2), \
            "transition_matrix must be 2x2"

        self.initial_regime = initial_regime
        self.regime = initial_regime

    # ---------------- Regime dynamics ---------------- #

    def _update_regime(self):
        """
        Evolve volatility regime as:
            regime_{t+1} ~ Categorical(transition_matrix[regime_t])
        """
        probs = self.transition_matrix[self.regime]
        self.regime = self.rng.choice([0, 1], p=probs)

    # ---------------- Jump sampling ---------------- #

    def _draw_jump(self):
        """Draw a single jump size J ~ N(jump_mean, jump_std)."""
        return self.rng.normal(self.jump_mean, self.jump_std)

    # ---------------- Price update ---------------- #

    def _update_price(self):
        """
        One-step update:

            1. Update regime (Markov chain)
            2. Choose sigma based on regime
            3. Diffusion term: mu*dt + sigma*sqrt(dt)*N(0,1)
            4. Jump term: J with prob jump_intensity*dt

        Finally:
            S <- S + diff + jump
        """
        # 1) Regime update
        self._update_regime()

        # 2) Regime-dependent volatility
        sigma = self.sigma_low if self.regime == 0 else self.sigma_high

        # 3) Diffusion part
        dW = self.rng.normal(0.0, np.sqrt(self.dt))
        dS_diff = self.mu * self.dt + sigma * dW

        # 4) Jump part
        if self.rng.uniform() < self.jump_intensity * self.dt:
            J = self._draw_jump()
        else:
            J = 0.0

        # Apply update
        self.S_prev = self.S
        self.S = self.S + dS_diff + J

    # ---------------- Reset ---------------- #

    def reset(self, *, seed=None, options=None):
        """
        Reset underlying base env and also reset regime to initial_regime.
        """
        obs, info = super().reset(seed=seed)
        self.regime = self.initial_regime
        return obs, info
