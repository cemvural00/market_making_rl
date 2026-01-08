import numpy as np
from agents.base_agent import BaseAgent


class FixedSpreadAgent(BaseAgent):
    """
    Heuristic 1: Fixed-spread symmetric quoting.

    The agent always quotes:

        bid = S - delta_fixed
        ask = S + delta_fixed

    by choosing a constant spread_factor and zero skew_factor.
    """

    def __init__(self, config=None):
        """
        config:
            - fixed_multiplier: float (default 1.0)
                delta = fixed_multiplier * base_delta
        """
        super().__init__(config)
        self.mult = self.config.get("fixed_multiplier", 1.0)

    def act(self, obs, info=None):
        """
        Converts desired fixed spread into spread_factor.
        """
        # We need base_delta from environment.
        # If env doesn't pass in info, user should include it in config.
        base_delta = None
        if info is not None and "base_delta" in info:
            base_delta = info["base_delta"]
        else:
            base_delta = self.config.get("base_delta", 1.0)

        # target delta = mult * base_delta
        # env delta = base_delta * (1 + 0.5 * spread_factor)
        # Solve for spread_factor:
        # spread_factor = 2*(mult - 1)
        spread_factor_raw = 2 * (self.mult - 1.0)
        spread_factor = float(np.clip(spread_factor_raw, -1.0, 1.0))

        skew_factor = 0.0  # symmetric quotes always

        return np.array([spread_factor, skew_factor], dtype=np.float32)
