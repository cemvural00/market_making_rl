import numpy as np
from agents.base_agent import BaseAgent

from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv

import torch
import torch.nn as nn


class LSTMPPOAgent(BaseAgent):
    """
    LSTM-based PPO agent for market-making environments.

    This uses RecurrentPPO from sb3-contrib, which supports LSTM
    memory across timesteps, making it suitable for:
        - price paths with temporal structure
        - LOB environments
        - synthetic jump/regime models
        - real-world sequential microstructure data

    This agent fits into the BaseAgent API:
        - act()
        - train()
        - save()
        - load()
    """

    def __init__(self, env, config=None):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Training environment. Should be a single env.

        config : dict
            LSTM + PPO hyperparameters.

            Common keys:
                total_timesteps: int (default 300_000)
                use_vec_env: bool (default False)
                learning_rate
                gamma
                n_steps
                batch_size
                tensorboard_log
                verbose

            LSTM-specific (policy_kwargs):
                policy_kwargs:
                    lstm_hidden_size: 128
                    n_lstm_layers: 1
                    shared_lstm: False
                    net_arch: [64, 64]
                    activation_fn: nn.ReLU
        """
        super().__init__(config)
        self.env = env

        # Agent-level config
        self.total_timesteps = self.config.pop("total_timesteps", 300_000)
        self.use_vec_env = self.config.pop("use_vec_env", False)
        
        # Early stopping configuration
        early_stopping_config = self.config.pop("early_stopping", None)
        self.early_stopping_enabled = early_stopping_config is not None and early_stopping_config.get("enabled", False)
        self.early_stopping_config = early_stopping_config if self.early_stopping_enabled else None

        # -------------------------
        # Default LSTM architecture
        # -------------------------
        default_policy_kwargs = {
            "lstm_hidden_size": 128,
            "n_lstm_layers": 1,
            "shared_lstm": False,
            "net_arch": [64, 64],  # post-LSTM MLP layers
            "activation_fn": nn.ReLU,
        }

        # Allow overrides from YAML
        user_policy_kwargs = self.config.pop("policy_kwargs", None)
        if user_policy_kwargs is not None:
            # Convert string activation_fn to class if needed (from YAML)
            if isinstance(user_policy_kwargs, dict) and "activation_fn" in user_policy_kwargs:
                if isinstance(user_policy_kwargs["activation_fn"], str):
                    if user_policy_kwargs["activation_fn"] == "ReLU":
                        user_policy_kwargs["activation_fn"] = nn.ReLU
                    elif user_policy_kwargs["activation_fn"] == "Tanh":
                        user_policy_kwargs["activation_fn"] = nn.Tanh
                    elif user_policy_kwargs["activation_fn"] == "Sigmoid":
                        user_policy_kwargs["activation_fn"] = nn.Sigmoid
            policy_kwargs = user_policy_kwargs
        else:
            policy_kwargs = default_policy_kwargs

        # Vectorize if needed
        if self.use_vec_env:
            self._sb3_env = DummyVecEnv([lambda: self.env])
        else:
            self._sb3_env = self.env

        # Remaining config keys passed to RecurrentPPO
        ppo_kwargs = self.config.copy()
        ppo_kwargs.setdefault("verbose", 0)
        ppo_kwargs["policy_kwargs"] = policy_kwargs
        
        # RecurrentPPO with MLP layers is faster on CPU than GPU/MPS
        # Similar to PPO, on-policy algorithms with MLP don't benefit from GPU
        device = "cpu"
        ppo_kwargs["device"] = device
        
        verbose_level = ppo_kwargs.get("verbose", 0)
        if verbose_level > 0:
            print(f"LSTMPPOAgent: Using device: {device} (CPU recommended for MLP policies)")

        self.model = RecurrentPPO(
            policy="MlpLstmPolicy",
            env=self._sb3_env,
            **ppo_kwargs
        )

        # Internal LSTM hidden states
        self._last_lstm_state = None
        self._last_done = False

    # ------------------------------------------------------
    #   ACTION METHOD (very important)
    # ------------------------------------------------------
    def act(self, obs, info=None):
        """
        Compute action using LSTM memory.
        RecurrentPPO requires passing hidden state in .predict().
        """
        # For RecurrentPPO, pass None as state on first call or episode start
        # The model will automatically initialize the state
        state = None if self._last_lstm_state is None else self._last_lstm_state
        episode_start = np.array([self._last_lstm_state is None], dtype=bool)
        
        action, lstm_state = self.model.predict(
            obs,
            deterministic=True,
            state=state,
            episode_start=episode_start,
        )
        
        # Update internal memory
        self._last_lstm_state = lstm_state
        self._last_done = np.array([False], dtype=bool)
        
        return np.array(action, dtype=np.float32)
    
    def reset_memory(self):
        """
        Reset LSTM hidden state (call at start of every new episode).
        This ensures each episode is evaluated independently.
        """
        self._last_lstm_state = None
        self._last_done = np.array([True], dtype=bool)

    # def act(self, obs, info=None):
    #     """
    #     Compute action using LSTM memory.
    #     RecurrentPPO requires passing hidden state in .predict().
    #     """
    #     if self._last_lstm_state is None:
    #         self._last_lstm_state = self.model.policy.recurrent_actor.initial_state(1)
    #         self._last_done = False
    #
    #     action, lstm_state = self.model.predict(
    #         obs,
    #         deterministic=True,
    #         state=self._last_lstm_state,
    #         episode_start=np.array([self._last_done], dtype=bool),
    #     )
    #
    #     # Update LSTM internal memory
    #     self._last_lstm_state = lstm_state
    #     self._last_done = False
    #
    #     return np.array(action, dtype=np.float32)

    # ------------------------------------------------------
    #   TRAINING
    # ------------------------------------------------------
    def train(self, env=None, total_timesteps=None):
        if env is not None:
            self.env = env
            self._sb3_env = DummyVecEnv([lambda: env]) if self.use_vec_env else env
            self.model.set_env(self._sb3_env)

        if total_timesteps is None:
            total_timesteps = self.total_timesteps

        # Reset LSTM state before training
        self._last_lstm_state = None
        self._last_done = False

        # Setup early stopping callback if enabled
        callback = None
        if self.early_stopping_enabled:
            from experiments.callbacks import RewardBasedEarlyStopping, LossBasedEarlyStopping
            es_config = self.early_stopping_config
            
            # Determine which type of early stopping to use
            monitor_type = es_config.get("monitor_type", "reward")  # "reward" or "loss"
            
            if monitor_type == "reward":
                # Create evaluation environment (copy of training env)
                eval_env = type(self.env)(
                    S0=getattr(self.env, 'S0', 100.0),
                    T=getattr(self.env, 'T', 1.0),
                    dt=getattr(self.env, 'dt', 0.0001),
                    A=getattr(self.env, 'A', 5.0),
                    k=getattr(self.env, 'k', 1.5),
                    base_delta=getattr(self.env, 'base_delta', 1.0),
                    max_inventory=getattr(self.env, 'max_inventory', 20),
                    inv_penalty=getattr(self.env, 'inv_penalty', 0.01),
                    seed=None
                )
                
                callback = RewardBasedEarlyStopping(
                    eval_env=eval_env,
                    monitor=es_config.get("monitor", "mean_reward"),
                    patience=es_config.get("patience", 10),
                    min_delta=es_config.get("min_delta", 0.0),
                    mode=es_config.get("mode", "max"),
                    best_model_save_path=es_config.get("best_model_path", None),
                    verbose=es_config.get("verbose", 1),
                    eval_freq=es_config.get("eval_freq", 2 * getattr(self.model, 'n_steps', 512)),  # Every 2 rollouts
                    n_eval_episodes=es_config.get("n_eval_episodes", 10),
                )
            else:
                # Fall back to loss-based early stopping
                n_steps = getattr(self.model, 'n_steps', 512)
                episode_length = getattr(self.env, 'n_steps', 10000)
                callback = LossBasedEarlyStopping(
                    patience=es_config.get("patience", 10),
                    n_steps=n_steps,
                    episode_length=episode_length,
                    min_delta=es_config.get("min_delta", 0.0),
                    verbose=es_config.get("verbose", 1)
                )
            
            if self.config.get("verbose", 0) > 0:
                monitor_name = es_config.get("monitor", "mean_reward" if monitor_type == "reward" else "loss")
                print(f"Early stopping enabled ({monitor_type}-based): "
                      f"patience={es_config.get('patience', 10)}, "
                      f"monitor={monitor_name}")

        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        
        # If using reward-based early stopping, load best model
        if self.early_stopping_enabled and callback is not None:
            if hasattr(callback, 'best_model_save_path') and callback.best_model_save_path is not None:
                import os
                from sb3_contrib import RecurrentPPO
                best_path = callback.best_model_save_path + ".zip"
                if os.path.exists(best_path):
                    self.model = RecurrentPPO.load(callback.best_model_save_path, env=self._sb3_env)
                    if self.config.get("verbose", 0) > 0:
                        print(f"Loaded best model from: {best_path}")

    # ------------------------------------------------------
    #   SAVING / LOADING
    # ------------------------------------------------------
    def save(self, path):
        self.model.save(path)

    def load(self, path, env=None):
        if env is not None:
            self.env = env
            self._sb3_env = DummyVecEnv([lambda: env]) if self.use_vec_env else env

        self.model = RecurrentPPO.load(path, env=self._sb3_env)
        self._last_lstm_state = None
        self._last_done = False
