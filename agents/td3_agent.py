import numpy as np
from .base_agent import BaseAgent

from stable_baselines3 import TD3
from stable_baselines3.common.vec_env import DummyVecEnv


class TD3Agent(BaseAgent):
    """
    Twin Delayed DDPG (TD3) RL market-making agent.
    
    TD3 is an off-policy algorithm that:
    - Uses deterministic policy (good for precise quote setting)
    - More stable than DDPG
    - Sample efficient (off-policy)
    - Well-suited for continuous control tasks
    
    This agent conforms to the BaseAgent interface:
        - act(obs, info=None)  -> action
        - train(env, total_timesteps=...)
        - save(path)
        - load(path, env=None)
    """

    def __init__(self, env, config=None):
        """
        Parameters
        ----------
        env : gymnasium.Env
            Training environment (single env, not vec env).
        config : dict
            Configuration dictionary:
            
            1) TD3Agent-level options:
                - "total_timesteps": int, default 300_000
                - "use_vec_env": bool, default False
            
            2) TD3-specific kwargs (forwarded to stable_baselines3.TD3):
                - "learning_rate": float, default 3e-4
                - "buffer_size": int, default 100_000
                - "learning_starts": int, default 100
                - "batch_size": int, default 256
                - "tau": float, default 0.005 (soft update coefficient)
                - "gamma": float, default 0.99
                - "train_freq": int or tuple, default 1
                - "gradient_steps": int, default 1
                - "policy_delay": int, default 2 (TD3-specific)
                - "target_policy_noise": float, default 0.2
                - "target_noise_clip": float, default 0.5
                - "verbose": int, default 0
        """
        super().__init__(config)

        self.env = env

        # Default agent-level settings
        self.total_timesteps = self.config.pop("total_timesteps", 300_000)
        self.use_vec_env = self.config.pop("use_vec_env", False)
        
        # Early stopping configuration
        early_stopping_config = self.config.pop("early_stopping", None)
        self.early_stopping_enabled = early_stopping_config is not None and early_stopping_config.get("enabled", False)
        self.early_stopping_config = early_stopping_config if self.early_stopping_enabled else None

        # Build env for TD3 (optionally vectorized)
        if self.use_vec_env:
            self._sb3_env = DummyVecEnv([lambda: self.env])
        else:
            self._sb3_env = self.env

        # All remaining config keys are passed directly to TD3
        td3_kwargs = self.config.copy()

        # Safe defaults if not provided
        if "verbose" not in td3_kwargs:
            td3_kwargs["verbose"] = 0

        self.model = TD3(
            policy="MlpPolicy",
            env=self._sb3_env,
            **td3_kwargs,
        )

    # ------------------------------------------------------
    #   Action interface
    # ------------------------------------------------------
    def act(self, obs, info=None):
        """
        Compute an action from the current observation.

        Parameters
        ----------
        obs : np.ndarray
            Current observation from env.reset() or env.step().
        info : dict or None
            (Unused here but kept for API compatibility.)

        Returns
        -------
        action : np.ndarray
            Action vector to be passed into env.step(action).
        """
        action, _ = self.model.predict(obs, deterministic=True)
        return np.array(action, dtype=np.float32)

    # ------------------------------------------------------
    #   Training
    # ------------------------------------------------------
    def train(self, env=None, total_timesteps=None):
        """
        Train the TD3 agent on the given environment.

        Parameters
        ----------
        env : gymnasium.Env or None
            If provided, replaces the current training env.
        total_timesteps : int or None
            If None, defaults to the value from config ("total_timesteps").
        """
        if env is not None:
            self.env = env
            if self.use_vec_env:
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
                    eval_freq=es_config.get("eval_freq", 10000),
                    n_eval_episodes=es_config.get("n_eval_episodes", 10),
                )
            else:
                # Fall back to loss-based early stopping
                # For TD3, use train_freq if available, otherwise default to 64 steps
                n_steps = getattr(self.model, 'train_freq', 64)
                if hasattr(n_steps, 'value'):  # train_freq might be a TrainFreq object
                    n_steps = n_steps.value
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
                    self.model = TD3.load(callback.best_model_save_path, env=self._sb3_env)
                    if self.config.get("verbose", 0) > 0:
                        print(f"Loaded best model from: {best_path}")

    # ------------------------------------------------------
    #   Save / load
    # ------------------------------------------------------
    def save(self, path):
        """
        Save TD3 model to disk.

        Parameters
        ----------
        path : str
            File path prefix (SB3 will add .zip).
        """
        self.model.save(path)

    def load(self, path, env=None):
        """
        Load TD3 model from disk.

        Parameters
        ----------
        path : str
            File path prefix used in save().
        env : gymnasium.Env or None
            If provided, set as the new env; else reuse current env.
        """
        if env is not None:
            self.env = env
            if self.use_vec_env:
                self._sb3_env = DummyVecEnv([lambda: self.env])
            else:
                self._sb3_env = self.env
        self.model = TD3.load(path, env=self._sb3_env)
