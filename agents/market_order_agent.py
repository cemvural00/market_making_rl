import numpy as np
from agents.base_agent import BaseAgent


class MarketOrderOnlyAgent(BaseAgent):
    """
    Heuristic 4: Market-order-only behavior.

    Since the environment does not support market orders directly,
    we emulate them by:
        - Minimizing spread (spread_factor = -1)
        - Maximally skewing toward inventory reduction
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.max_inv = self.config.get("max_inventory", 20)

    def act(self, obs, info=None):
        norm_time, S_norm, dS, q_norm = obs[:4]
        q = q_norm * self.max_inv

        # Collapse spread
        spread_factor = -1.0

        if q > 0:
            # long → want to sell → push quotes downward
            skew_factor = 1.0
        elif q < 0:
            # short → want to buy → push quotes upward
            skew_factor = -1.0
        else:
            # no inventory → no need to act
            spread_factor = -1.0
            skew_factor = 0.0

        return np.array([spread_factor, skew_factor], dtype=np.float32)
