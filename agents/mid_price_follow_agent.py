import numpy as np
from agents.base_agent import BaseAgent


class MidPriceFollowAgent(BaseAgent):
    """
    Heuristic 6: Mid-price-following agent (signal-based).

    Idea:
        - Use short-term trend dS as a directional signal.
        - If dS > 0 (uptrend) → skew to build long inventory.
            → bid more aggressive, ask less aggressive
        - If dS < 0 (downtrend) → skew to build short inventory.
            → ask more aggressive, bid less aggressive

    Spread stays constant (for simplicity).
    """

    def __init__(self, config=None):
        """
        config:
            - trend_sensitivity: float (default 0.5)
                Strength of reaction to dS.
            - max_skew: float in (0,1], default 1.0
                Max allowed skew before clipping.
        """
        super().__init__(config)
        self.beta = self.config.get("trend_sensitivity", 0.5)
        self.max_skew = self.config.get("max_skew", 1.0)

    def act(self, obs, info=None):
        """
        obs = [norm_time, S_norm, dS, q_norm]
        """
        norm_time, S_norm, dS, q_norm = obs

        # Spread stays constant
        spread_factor = 0.0

        # Trend signal: skew into direction of expected movement
        #
        # If dS > 0:
        #   skew_factor > 0 → pushes quotes downward → encourages buying → builds long position
        # If dS < 0:
        #   skew_factor < 0 → pushes quotes upward → encourages selling → builds short position
        #
        skew_raw = self.beta * np.tanh(dS)  # smooth bounded signal
        skew_factor = float(np.clip(skew_raw, -self.max_skew, self.max_skew))

        return np.array([spread_factor, skew_factor], dtype=np.float32)
