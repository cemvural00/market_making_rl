import numpy as np
from agents.base_agent import BaseAgent


class InventoryShiftAgent(BaseAgent):
    """
    Heuristic 3: Inventory-based quote shifting.

    Spread remains fixed (base_delta), but skew_term shifts quotes:
        skew_factor ∝ q
    """

    def __init__(self, config=None):
        """
        config:
            - beta: skew sensitivity (default 0.05)
            - max_inventory: used to de-normalize q_norm
        """
        super().__init__(config)
        self.beta = self.config.get("beta", 0.05)
        self.max_inv = self.config.get("max_inventory", 20)

    def act(self, obs, info=None):
        norm_time, S_norm, dS, q_norm = obs[:4]
        q = q_norm * self.max_inv

        # Keep spread fixed
        spread_factor = 0.0

        # Skew factor ∝ q
        skew_factor_raw = self.beta * q
        skew_factor = float(np.clip(skew_factor_raw, -1, 1))

        return np.array([spread_factor, skew_factor], dtype=np.float32)
