class BaseAgent:
    """
    Abstract base class for all market-making agents.

    Every agent must implement:
        - act(obs, info=None) → action

    Optional methods (for RL agents):
        - train(env)
        - save(path)
        - load(path)
    """

    def __init__(self, config=None):
        """
        Parameters
        ----------
        config : dict or None
            Hyperparameters or settings for the agent.
        """
        self.config = config or {}

    # ------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------
    def act(self, obs, info=None):
        """
        Compute an action given the current observation.

        Must be implemented by subclasses.

        Returns
        -------
        action : np.ndarray
            Environment action vector.
        """
        raise NotImplementedError("act() must be implemented by child agent classes.")

    # ------------------------------------------------------
    # Optional (used for RL agents)
    # ------------------------------------------------------
    def train(self, env):
        """
        Train the agent on the given environment.

        Default: agents that cannot train do nothing.
        """
        raise NotImplementedError("train() must be implemented by RL agents.")

    def save(self, path):
        """
        Save agent parameters to disk.
        """
        raise NotImplementedError("save() must be implemented by RL agents.")

    def load(self, path):
        """
        Load agent parameters from disk.
        """
        raise NotImplementedError("load() must be implemented by RL agents.")
