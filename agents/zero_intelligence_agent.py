import numpy as np
from agents.base_agent import BaseAgent


class ZeroIntelligenceAgent(BaseAgent):
    """
    Heuristic 8: Zero-Intelligence Poisson-like quoting.

    Each action dimension is randomly drawn from Uniform[-1, 1].
    No reference to market state. Pure noise.
    """

    def __init__(self, config=None):
        super().__init__(config)

    def act(self, obs, info=None):
        spread_factor = np.random.uniform(-1, 1)
        skew_factor = np.random.uniform(-1, 1)
        return np.array([spread_factor, skew_factor], dtype=np.float32)
