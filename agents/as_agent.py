import numpy as np
from .base_agent import BaseAgent


class ASSimpleHeuristicAgent(BaseAgent):
    """
    Simple AS-inspired heuristic agent.

    This is essentially the earlier notebook policy:
      - spread depends on time-to-maturity
      - skew depends linearly on inventory

    It is NOT the closed-form optimal AS control, but a
    interpretable benchmark that behaves "AS-like".
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict
            - "gamma": inventory sensitivity (default 0.1)
            - "min_spread_factor": min spread_factor (default 0.2)
            - "max_spread_factor": max spread_factor (default 0.8)
            - "max_inventory": used to de-normalize q if env doesn’t provide
        """
        super().__init__(config)

        self.gamma = self.config.get("gamma", 0.1)
        self.min_spread_factor = self.config.get("min_spread_factor", 0.2)
        self.max_spread_factor = self.config.get("max_spread_factor", 0.8)
        self.default_max_inventory = self.config.get("max_inventory", 20)

    def act(self, obs, info=None):
        """
        obs: [norm_time, S_norm, dS, q_norm]
        """
        norm_time, S_norm, dS, q_norm = obs

        time_left = 1.0 - float(norm_time)

        # Spread factor: wide early, narrow near maturity
        spread_factor = (
                self.min_spread_factor
                + (self.max_spread_factor - self.min_spread_factor) * time_left
        )
        spread_factor = float(np.clip(spread_factor, -1.0, 1.0))

        max_inv = (
            info.get("max_inventory")
            if info is not None and "max_inventory" in info
            else self.default_max_inventory
        )
        q = float(q_norm) * max_inv

        skew_factor = float(np.clip(self.gamma * q, -1.0, 1.0))

        return np.array([spread_factor, skew_factor], dtype=np.float32)


class ASClosedFormAgent(BaseAgent):
    """
    Avellaneda–Stoikov closed-form quoting agent (approximate).

    Implements the standard AS model under the usual assumptions:

        - Midprice S follows Brownian motion with volatility sigma
        - Order arrivals have intensity λ(δ) = A exp(-k δ)
        - Exponential utility with risk aversion gamma
        - Finite horizon [0, T]

    The AS solution gives:
        - an optimal half-spread δ*(t)
        - a reservation price r(q, t) that depends on inventory q

    We do not directly set bid/ask, but instead map δ* and
    inventory-induced reservation price shift into the action space
        action = [spread_factor, skew_factor] ∈ [-1, 1]^2
    which your environments interpret as:

        delta = base_delta * (1 + 0.5 * spread_factor)
        skew_term = skew_factor * 0.5 * delta

        bid = S - delta - skew_term
        ask = S + delta - skew_term

    We try to match:

        delta ≈ δ*
        skew_term ≈ inventory_adjustment = γ σ^2 (T - t) q   (up to constants)
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict
            Required / important keys:

            - "gamma": float
                Risk aversion parameter γ (AS utility).
            - "sigma": float
                Volatility σ of the mid-price (per unit time, consistent with T).
            - "k": float
                Order arrival decay parameter k in λ(δ) = A exp(-k δ).
            - "base_delta": float
                The base half-spread used by the environment.
                Must match env.base_delta for best consistency.
            - "max_inventory": int
                Used to de-normalize inventory from q_norm.

            Optional:
            - "T": float
                Effective horizon in AS time units. If None, we assume
                that env's normalized time runs from 0 to 1 and T=1.
        """
        super().__init__(config)

        # Core AS parameters
        self.gamma = float(self.config.get("gamma", 0.1))
        self.sigma = float(self.config.get("sigma", 2.0))
        self.k = float(self.config.get("k", 1.5))  # intensity slope in λ(δ)
        self.base_delta = float(self.config.get("base_delta", 1.0))
        self.max_inventory = float(self.config.get("max_inventory", 20.0))

        # Horizon in "AS time units" – by default 1, since env uses norm_time in [0,1]
        self.T = float(self.config.get("T", 1.0))

    # -------------------- Core AS formulas -------------------- #

    def _delta_star(self, tau):
        """
        AS optimal half-spread δ*(t) under
            λ(δ) = A exp(-k δ)

        Standard formula (see Avellaneda–Stoikov 2008):

            δ* = (1/γ) * ln(1 + γ/k) + 0.5 * γ σ^2 (T - t)

        where:
            γ = risk aversion
            σ = mid-price volatility
            k = intensity slope
            τ = T - t
        """
        term1 = (1.0 / self.gamma) * np.log(1.0 + self.gamma / self.k)
        term2 = 0.5 * self.gamma * (self.sigma ** 2) * tau
        return term1 + term2

    def _inventory_adjustment(self, q, tau):
        """
        Inventory-induced reservation price shift.

        In the AS model, the reservation price r(q,t) is:

            r(q,t) = S - γ σ^2 (T - t) q      (up to model conventions)

        So the shift away from midprice S is:

            inv_adj = γ σ^2 (T - t) q

        We absorb this into the skew term of the quoting rule.
        """
        return self.gamma * (self.sigma ** 2) * tau * q

    # -------------------- Action mapping -------------------- #

    def act(self, obs, info=None):
        """
        Map current state into AS optimal-style action.

        obs: [norm_time, S_norm, dS, q_norm]
        info: optional dict (not strictly needed here)

        Returns
        -------
        action : np.ndarray
            [spread_factor, skew_factor] in [-1, 1]^2
        """
        norm_time, S_norm, dS, q_norm = obs

        # Continuous AS time left: τ = T - t
        # Here norm_time ∈ [0,1] is t/T, so t = norm_time * T → τ = T*(1 - norm_time)
        tau = max(1e-8, self.T * (1.0 - float(norm_time)))

        # Denormalize inventory
        q = float(q_norm) * self.max_inventory

        # 1) Optimal half-spread
        delta_star = self._delta_star(tau)  # target half-spread

        # 2) Inventory-related reservation price shift
        inv_adj = self._inventory_adjustment(q, tau)  # shift from midprice

        # Our env interprets action as:
        #   delta_env = base_delta * (1 + 0.5 * spread_factor)
        #   skew_term = skew_factor * 0.5 * delta_env
        #
        # We want (approximately):
        #   delta_env ≈ δ*
        #   skew_term ≈ inv_adj
        #
        # => spread_factor ≈ 2*(δ*/base_delta - 1)
        #    skew_factor  ≈ 2*inv_adj / delta_env

        # Avoid degenerate base_delta
        base_delta = max(self.base_delta, 1e-6)

        # Compute spread_factor (unclipped)
        spread_factor_raw = 2.0 * (delta_star / base_delta - 1.0)

        # For skew_factor we need the actual delta_env that will result.
        # After clipping, delta_env will be:
        #   delta_env = base_delta * (1 + 0.5 * spread_factor_clipped)
        # so we do a two-step process: approximate via raw, then refine.

        # First clip spread factor
        spread_factor = float(np.clip(spread_factor_raw, -1.0, 1.0))

        # Resulting half-spread in the env:
        delta_env = base_delta * (1.0 + 0.5 * spread_factor)
        delta_env = max(delta_env, 1e-6)

        # Now choose skew_factor so that skew_term ≈ inv_adj:
        #   skew_term = skew_factor * 0.5 * delta_env ≈ inv_adj
        # => skew_factor ≈ 2 * inv_adj / delta_env
        skew_factor_raw = 2.0 * inv_adj / delta_env
        skew_factor = float(np.clip(skew_factor_raw, -1.0, 1.0))

        return np.array([spread_factor, skew_factor], dtype=np.float32)
