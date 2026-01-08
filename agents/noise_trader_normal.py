import numpy as np
from agents.base_agent import BaseAgent


class NoiseTraderNormal(BaseAgent):
    """
    Noise agent drawing spread/skew from Gaussian distributions.
    """

    def __init__(self, config=None):
        """
        config:
            - spread_std: float, default 0.3
            - skew_std: float,   default 0.3
        """
        super().__init__(config)
        self.s_spread = self.config.get("spread_std", 0.3)
        self.s_skew   = self.config.get("skew_std", 0.3)

    def act(self, obs, info=None):
        spread = np.random.normal(0, self.s_spread)
        skew   = np.random.normal(0, self.s_skew)

        return np.array([
            float(np.clip(spread, -1, 1)),
            float(np.clip(skew,   -1,  1))
        ], dtype=np.float32)
