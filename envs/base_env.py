import numpy as np
import gymnasium as gym
from gymnasium import spaces
from abc import ABC, abstractmethod


class MarketMakingBaseEnv(gym.Env, ABC):
    """
    Base class for all synthetic market-making environments.

    This class contains:
        - Common state variables
        - Action interpretation (spread_factor, skew_factor)
        - Bid/ask construction
        - Poisson fill model λ(δ) = A exp(-k δ)
        - Inventory & cash accounting
        - Reward = ΔPnL - inv_penalty * q^2 * dt
        - Gymnasium reset() and step() logic

    Child classes ONLY implement one method:
        - _update_price()

    Examples of child envs:
        BMEnv  : dS = sigma dW
        OUEnv  : dS = kappa (mu - S) dt + sigma dW
        JumpEnv: dS = mu dt + sigma dW + J
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        S0=100.0,
        T=1.0,
        dt=0.0001,
        A=5.0,
        k=1.5,
        base_delta=1.0,
        max_inventory=20,
        inv_penalty=0.01,
        seed=None,
    ):
        super().__init__()

        # ----- Market & simulation params -----
        self.S0 = S0
        self.T = T
        self.dt = dt
        self.n_steps = int(T / dt)

        # ----- Fill intensity params -----
        self.A = A
        self.k = k

        # ----- Execution & inventory -----
        self.base_delta = base_delta
        self.max_inventory = max_inventory
        self.inv_penalty = inv_penalty

        # RNG
        self.rng = np.random.default_rng(seed)

        # ----- Action space -----
        # [spread_factor, skew_factor] in [-1, 1]^2
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ----- Observation space -----
        # [normalized_time, S/S0, dS, q/max_inventory]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -np.inf, -1.0], dtype=np.float32),
            high=np.array([1.0, np.inf,  np.inf,   1.0], dtype=np.float32),
            dtype=np.float32,
        )

        # ----- Internal state -----
        self.t_index = 0
        self.S = None
        self.S_prev = None
        self.q = None
        self.X = None
        self.done = False

    # ============================================================
    # ===============  ABSTRACT PRICE UPDATE  =====================
    # ============================================================

    @abstractmethod
    def _update_price(self):
        """
        Child environments implement this:
            BMEnv:  self.S = self.S + sigma * dW
            OUEnv:  self.S = self.S + kappa*(mu - S)*dt + sigma*dW
            JumpEnv: self.S = self.S + mu*dt + sigma*dW + J

        Must update:
            - self.S_prev
            - self.S
        """
        pass

    # ============================================================
    # ===================== OBSERVATION ===========================
    # ============================================================

    def _get_obs(self):
        norm_time = self.t_index / self.n_steps
        S_norm = self.S / self.S0
        dS = self.S - self.S_prev
        q_norm = self.q / self.max_inventory
        return np.array([norm_time, S_norm, dS, q_norm], dtype=np.float32)

    # ============================================================
    # ===================== FILL MODEL ============================
    # ============================================================

    def _intensity(self, delta):
        """λ(δ) = A exp(-k δ) for δ >= 0."""
        delta = np.maximum(delta, 0.0)
        return self.A * np.exp(-self.k * delta)

    # ============================================================
    # ======================== RESET ==============================
    # ============================================================

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.rng = np.random.default_rng(seed)

        self.t_index = 0
        self.S = self.S0
        self.S_prev = self.S0
        self.q = 0
        self.X = 0.0
        self.done = False

        return self._get_obs(), {}

    # ============================================================
    # ========================= STEP ==============================
    # ============================================================

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        # ----- Interpret action -----
        spread_factor = float(np.clip(action[0], -1.0, 1.0))
        skew_factor   = float(np.clip(action[1], -1.0, 1.0))

        # half-spread
        delta = self.base_delta * (1.0 + 0.5 * spread_factor)
        delta = max(delta, 0.01)

        # skew term
        skew = skew_factor * 0.5 * delta

        bid = self.S - delta - skew
        ask = self.S + delta - skew

        # ----- Fill intensities -----
        lam_b = self._intensity(self.S - bid)
        lam_a = self._intensity(ask - self.S)

        # Bernoulli approx
        filled_bid = self.rng.uniform() < lam_b * self.dt
        filled_ask = self.rng.uniform() < lam_a * self.dt

        # Save old values
        X_old = self.X
        q_old = self.q
        S_old = self.S

        # ----- Inventory & cash updates -----
        if filled_bid and self.q < self.max_inventory:
            self.q += 1
            self.X -= bid

        if filled_ask and self.q > -self.max_inventory:
            self.q -= 1
            self.X += ask

        # ----- Price update (child defines logic) -----
        self._update_price()

        # ----- Reward -----
        pnl_old = X_old + q_old * S_old
        pnl_new = self.X + self.q * self.S
        pnl_change = pnl_new - pnl_old

        reward = pnl_change - self.inv_penalty * (self.q ** 2) * self.dt

        # ----- Time update -----
        self.t_index += 1
        if self.t_index >= self.n_steps:
            self.done = True

        obs = self._get_obs()
        info = {"bid": bid, "ask": ask, "pnl": pnl_new, "inventory": self.q}

        return obs, float(reward), self.done, False, info
