import numpy as np
from agents.base_agent import BaseAgent

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

import torch.nn as nn


class LSTMPPOAgent(BaseAgent):
    """
    LSTM-based PPO agent for market-making environments.

    Uses RecurrentPPO from sb3-contrib to incorporate temporal memory.
    Suitable for:
        - synthetic price processes with jumps/regimes
        - OU / GBM environments
        - sparse LOB state representations
        - full-depth LOB event streams

    Integrates with the standard BaseAgent API:
        - act()
        - train()
        - save()
        - load()
        - reset_memory()
    """

    def __init__(self, env, config=None):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Environment for training and inference.

        config : dict
            LSTM + PPO hyperparameters, typically loaded from YAML.

            Common keys:
                total_timesteps: int (default 300_000)
                use_vec_env: bool
                learning_rate
                gamma
                n_steps
                batch_size
                policy_kwargs:
                    lstm_hidden_size
                    n_lstm_layers
                    net_arch
        """
        super().__init__(config)
        self.env = env

        # ===============================
        #   CONFIG EXTRACTION
        # ===============================
        self.total_timesteps = self.config.pop("total_timesteps", 300_000)
        self.use_vec_env = self.config.pop("use_vec_env", False)

        # Default LSTM architecture
        default_policy_kwargs = {
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "net_arch": [64, 64],
            "activation_fn": nn.ReLU,
        }

        policy_kwargs = self.config.pop("policy_kwargs", default_policy_kwargs)

        # Vectorize env if needed
        if self.use_vec_env:
            self._sb3_env = DummyVecEnv([lambda: self.env])
        else:
            self._sb3_env = self.env

        # Remaining config gets passed to PPO
        ppo_kwargs = self.config.copy()
        ppo_kwargs.setdefault("verbose", 0)
        ppo_kwargs["policy_kwargs"] = policy_kwargs

        # ===============================
        #   CREATE MODEL
        # ===============================
        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=self._sb3_env,
            **ppo_kwargs
        )

        # LSTM internal state
        self._last_lstm_state = None
        self._last_episode_start = np.array([True], dtype=bool)

    # ==========================================================
    #   MEMORY CONTROLS
    # ==========================================================
    def reset_memory(self):
        """Reset LSTM hidden state (call at start of every new episode)."""
        self._last_lstm_state = None
        self._last_episode_start = np.array([True], dtype=bool)

    # ==========================================================
    #   ACTION COMPUTATION
    # ==========================================================
    def act(self, obs, info=None):
        """
        Compute action using the LSTM memory.
        RecurrentPPO's predict() requires:
            - obs
            - previous lstm_state
            - episode_start flag to reset hidden state
        """

        # Initialize hidden state on first step
        if self._last_lstm_state is None:
            self._last_lstm_state = self.model.policy.get_initial_state()
            self._last_episode_start = np.array([True], dtype=bool)

        # Predict with stored state
        action, lstm_state = self.model.predict(
            obs,
            deterministic=True,
            state=self._last_lstm_state,
            episode_start=self._last_episode_start,
        )

        # Update memory
        self._last_lstm_state = lstm_state
        self._last_episode_start = np.array([False], dtype=bool)

        return np.array(action, dtype=np.float32)

    # ==========================================================
    #   TRAINING
    # ==========================================================
    def train(self, env=None, total_timesteps=None):
        if env is not None:
            self.env = env
            self._sb3_env = DummyVecEnv([lambda: env]) if self.use_vec_env else env
            self.model.set_env(self._sb3_env)

        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        # Reset LSTM state before training
        self.reset_memory()

        self.model.learn(total_timesteps=total_timesteps)

    # ==========================================================
    #   SAVE / LOAD
    # ==========================================================
    def save(self, path):
        self.model.save(path)

    def load(self, path, env=None):
        if env is not None:
            self.env = env
            self._sb3_env = DummyVecEnv([lambda: env]) if self.use_vec_env else env

        self.model = RecurrentPPO.load(path, env=self._sb3_env)
        self.reset_memory()
