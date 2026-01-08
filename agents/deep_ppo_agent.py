import numpy as np
from agents.base_agent import BaseAgent

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import torch.nn as nn


class DeepPPOAgent(BaseAgent):
    """
    Deep PPO-based RL market-making agent.

    This is similar to PPOAgent but:
      - uses a deeper neural network by default (e.g. [256, 256, 256])
      - exposes policy_kwargs for easy customization

    It conforms to the BaseAgent interface:
        - act(obs, info=None)
        - train(env=None, total_timesteps=None)
        - save(path)
        - load(path, env=None)
    """

    def __init__(self, env, config=None):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Training environment (single env).
        config : dict
            Configuration dictionary.

            DeepPPOAgent-level keys:
                - "total_timesteps": int, default 300_000
                - "use_vec_env": bool, default False

            PPO-specific keys (forwarded to PPO):
                - "learning_rate"
                - "gamma"
                - "n_steps"
                - "batch_size"
                - "ent_coef"
                - "vf_coef"
                - "tensorboard_log"
                - "verbose"
                - "policy_kwargs"  (optional, overrides default deep net)

            If "policy_kwargs" is not provided, we use:
                policy_kwargs = {
                    "net_arch": [256, 256, 256],
                    "activation_fn": nn.ReLU,
                }
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

        # Policy kwargs: deep network by default
        default_policy_kwargs = {
            "net_arch": [256, 256, 256],
            "activation_fn": nn.ReLU,
        }
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

        # Wrap env if requested
        if self.use_vec_env:
            self._sb3_env = DummyVecEnv([lambda: self.env])
        else:
            self._sb3_env = self.env

        # Fill remaining PPO kwargs
        ppo_kwargs = self.config.copy()
        ppo_kwargs.setdefault("verbose", 0)
        ppo_kwargs["policy_kwargs"] = policy_kwargs

        self.model = PPO(
            policy="MlpPolicy",
            env=self._sb3_env,
            **ppo_kwargs,
        )

    # ------------------------------------------------------
    #   Action interface
    # ------------------------------------------------------
    def act(self, obs, info=None):
        """
        Compute an action given the current observation.

        Parameters
        ----------
        obs : np.ndarray
        info : dict or None

        Returns
        -------
        action : np.ndarray (float32)
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return np.array(action, dtype=np.float32)

    # ------------------------------------------------------
    #   Training
    # ------------------------------------------------------
    def train(self, env=None, total_timesteps=None):
        """
        Train the agent.

        Parameters
        ----------
        env : gymnasium.Env or None
            If provided, replace the current env.
        total_timesteps : int or None
            If None, fall back to config["total_timesteps"].
        """
        if env is not None:
            self.env = env
            if self.use_vec_env:
                from stable_baselines3.common.vec_env import DummyVecEnv
                self._sb3_env = DummyVecEnv([lambda: self.env])
            else:
                self._sb3_env = self.env
            self.model.set_env(self._sb3_env)

        if total_timesteps is None:
            total_timesteps = self.total_timesteps

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
                    eval_freq=es_config.get("eval_freq", max(getattr(self.model, 'n_steps', 1024), 10000)),
                    n_eval_episodes=es_config.get("n_eval_episodes", 10),
                )
            else:
                # Fall back to loss-based early stopping
                n_steps = getattr(self.model, 'n_steps', 1024)
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
                best_path = callback.best_model_save_path + ".zip"
                if os.path.exists(best_path):
                    self.model = PPO.load(callback.best_model_save_path, env=self._sb3_env)
                    if self.config.get("verbose", 0) > 0:
                        print(f"Loaded best model from: {best_path}")

    # ------------------------------------------------------
    #   Persistence
    # ------------------------------------------------------
    def save(self, path):
        """
        Save model to disk.
        """
        self.model.save(path)

    def load(self, path, env=None):
        """
        Load model from disk.

        Parameters
        ----------
        path : str
            Path used when saving.
        env : gymnasium.Env or None
            If provided, set as environment for further training.
        """
        if env is not None:
            self.env = env
            if self.use_vec_env:
                from stable_baselines3.common.vec_env import DummyVecEnv
                self._sb3_env = DummyVecEnv([lambda: self.env])
            else:
                self._sb3_env = self.env

        self.model = PPO.load(path, env=self._sb3_env)
