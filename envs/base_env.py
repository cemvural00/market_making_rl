from collections import deque

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

    # GBM subclasses set this to True so lag returns normalise by S_t instead of S_0
    _normalize_lags_by_current_price: bool = False

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

        # ----- Observation space (16 features) -----
        self.observation_space = spaces.Box(
            low=-np.inf * np.ones(16, dtype=np.float32),
            high= np.inf * np.ones(16, dtype=np.float32),
            dtype=np.float32,
        )

        # ----- Internal state -----
        self.t_index = 0
        self.S = None
        self.S_prev = None
        self.q = None
        self.X = None
        self.done = False

        # sigma_0 for vol normalisation — set lazily in reset() after child __init__ runs
        self.sigma_0 = None

        # Ring buffers — initialised in reset()
        self.price_buffer = deque([S0] * 201, maxlen=201)
        self.q_buffer = deque([0] * 21, maxlen=21)
        self.bid_fill_buffer = deque([0] * 100, maxlen=100)
        self.ask_fill_buffer = deque([0] * 100, maxlen=100)
        self.t_since_last_bid_fill = 0
        self.t_since_last_ask_fill = 0

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
        # --- Existing 4 features ---
        norm_time = self.t_index / self.n_steps
        S_norm    = self.S / self.S0
        dS        = self.S - self.S_prev
        q_norm    = self.q / self.max_inventory

        # --- Price array from buffer (len 201) ---
        price_arr = np.asarray(self.price_buffer, dtype=np.float64)

        # --- Features 5-6: Realized volatility ---
        dS_short = np.diff(price_arr[-51:])   # 50 squared price increments
        dS_long  = np.diff(price_arr)          # 200 squared price increments
        rv_short = np.sqrt(np.mean(dS_short ** 2))
        rv_long  = np.sqrt(np.mean(dS_long  ** 2))
        sigma_0  = self.sigma_0 if self.sigma_0 is not None else 1.0
        rv_short_norm = rv_short / (sigma_0 + 1e-8)
        rv_long_norm  = rv_long  / (sigma_0 + 1e-8)

        # --- Feature 7: Volatility ratio (dimensionless regime proxy) ---
        vol_ratio = rv_short / (rv_long + 1e-8)

        # --- Feature 8: Fill imbalance over last 100 steps ---
        bid_sum = float(np.sum(self.bid_fill_buffer))
        ask_sum = float(np.sum(self.ask_fill_buffer))
        fill_imbalance = (bid_sum - ask_sum) / (bid_sum + ask_sum + 1e-8)

        # --- Features 9-10: Time since last fill (normalised to [0, 1]) ---
        time_since_bid = self.t_since_last_bid_fill / self.n_steps
        time_since_ask = self.t_since_last_ask_fill / self.n_steps

        # --- Features 11-13: Lagged returns ---
        # GBM: normalise by current S (multiplicative process); others: normalise by S0
        lag_denom = self.S if self._normalize_lags_by_current_price else self.S0
        lag_ret_5   = (self.S - price_arr[-6])   / (lag_denom + 1e-8)
        lag_ret_20  = (self.S - price_arr[-21])  / (lag_denom + 1e-8)
        lag_ret_100 = (self.S - price_arr[-101]) / (lag_denom + 1e-8)

        # --- Feature 14: Inventory × time-remaining (AS reservation price term) ---
        inv_time = q_norm * (1.0 - norm_time)

        # --- Feature 15: Inventory velocity over last 20 steps ---
        q_arr = np.asarray(self.q_buffer, dtype=np.float64)   # len 21
        inv_velocity = (self.q - q_arr[0]) / self.max_inventory

        # --- Feature 16: Jump indicator ---
        # rv_long is already the per-step RMS (≈ sigma * sqrt(dt)), so the
        # 3-sigma threshold for a single price increment is 3 * rv_long directly.
        jump_flag = 1.0 if abs(dS) > 3.0 * rv_long else 0.0

        return np.array([
            norm_time, S_norm, dS, q_norm,
            rv_short_norm, rv_long_norm, vol_ratio,
            fill_imbalance, time_since_bid, time_since_ask,
            lag_ret_5, lag_ret_20, lag_ret_100,
            inv_time, inv_velocity, jump_flag,
        ], dtype=np.float32)

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

        # Detect sigma_0 once — child __init__ sets self.sigma / self.sigma_low
        # before or after super().__init__, so reset() is the safe detection point.
        if self.sigma_0 is None:
            if hasattr(self, 'sigma'):
                self.sigma_0 = self.sigma
            elif hasattr(self, 'sigma_low'):
                self.sigma_0 = self.sigma_low
            else:
                self.sigma_0 = 1.0

        # Re-initialise ring buffers
        self.price_buffer = deque([self.S0] * 201, maxlen=201)
        self.q_buffer = deque([0] * 21, maxlen=21)
        self.bid_fill_buffer = deque([0] * 100, maxlen=100)
        self.ask_fill_buffer = deque([0] * 100, maxlen=100)
        self.t_since_last_bid_fill = 0
        self.t_since_last_ask_fill = 0

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
        actual_bid_fill = False
        if filled_bid and self.q < self.max_inventory:
            self.q += 1
            self.X -= bid
            actual_bid_fill = True

        actual_ask_fill = False
        if filled_ask and self.q > -self.max_inventory:
            self.q -= 1
            self.X += ask
            actual_ask_fill = True

        # ----- Update fill buffers & time-since-fill counters -----
        self.bid_fill_buffer.append(int(actual_bid_fill))
        self.ask_fill_buffer.append(int(actual_ask_fill))
        self.t_since_last_bid_fill = 0 if actual_bid_fill else self.t_since_last_bid_fill + 1
        self.t_since_last_ask_fill = 0 if actual_ask_fill else self.t_since_last_ask_fill + 1

        # ----- Price update (child defines logic) -----
        self._update_price()

        # ----- Update price & inventory buffers -----
        self.price_buffer.append(self.S)
        self.q_buffer.append(self.q)

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
