import numpy as np
from agents.base_agent import BaseAgent


class InventorySpreadScalerAgent(BaseAgent):
    """
    Heuristic 2: Spread increases with |inventory|.

    Implements:
        delta = base_delta * (1 + alpha * |q|)
    """

    def __init__(self, config=None):
        """
        config:
            - alpha: float (spread sensitivity) default 0.05
            - max_inventory: int (default 20)
        """
        super().__init__(config)
        self.alpha = self.config.get("alpha", 0.05)
        self.max_inv = self.config.get("max_inventory", 20)

    def act(self, obs, info=None):
        """
        obs = [norm_time, S_norm, dS, q_norm]
        """
        norm_time, S_norm, dS, q_norm = obs[:4]
        q = q_norm * self.max_inv

        # retrieve base_delta
        base_delta = None
        if info is not None and "base_delta" in info:
            base_delta = info["base_delta"]
        else:
            base_delta = self.config.get("base_delta", 1.0)

        delta_target = base_delta * (1 + self.alpha * abs(q))

        # convert to spread_factor
        # spread_factor = 2 * (delta_target / base_delta - 1)
        spread_factor_raw = 2 * (delta_target / base_delta - 1)
        spread_factor = float(np.clip(spread_factor_raw, -1, 1))

        skew_factor = 0.0  # symmetric widening

        return np.array([spread_factor, skew_factor], dtype=np.float32)
