import numpy as np
from agents.base_agent import BaseAgent


class NoiseTraderUniform(BaseAgent):
    """
    Noise agent drawing random spread/skew from uniform bands.
    """

    def __init__(self, config=None):
        """
        config:
            - spread_range: float, default 0.5   → U(-0.5, 0.5)
            - skew_range:   float, default 0.5
        """
        super().__init__(config)
        self.r_spread = self.config.get("spread_range", 0.5)
        self.r_skew   = self.config.get("skew_range", 0.5)

    def act(self, obs, info=None):
        spread = np.random.uniform(-self.r_spread, self.r_spread)
        skew   = np.random.uniform(-self.r_skew,   self.r_skew)

        spread = np.clip(spread, -1, 1)
        skew   = np.clip(skew,   -1, 1)

        return np.array([spread, skew], dtype=np.float32)
