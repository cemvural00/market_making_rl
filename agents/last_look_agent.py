import numpy as np
from agents.base_agent import BaseAgent


class LastLookAgent(BaseAgent):
    """
    Heuristic 5: Last-look quoting based on short-term momentum.

    Logic:
        - If price is going up (dS > 0):
            bid becomes more aggressive (narrower spread)
            ask becomes more conservative (wider spread)
        - If price is going down:
            reverse
    """

    def __init__(self, config=None):
        """
        config:
            - trend_sensitivity: float, default 0.5
                Controls how strongly the agent reacts to dS.
        """
        super().__init__(config)
        self.beta = self.config.get("trend_sensitivity", 0.5)

    def act(self, obs, info=None):
        norm_time, S_norm, dS, q_norm = obs

        # spread stays mostly normal
        spread_factor = 0.0

        # skew based on recent trend dS
        # Positive trend: skew_factor > 0 → pushes quotes downward
        skew_factor_raw = -self.beta * np.tanh(dS)
        skew_factor = float(np.clip(skew_factor_raw, -1, 1))

        return np.array([spread_factor, skew_factor], dtype=np.float32)
