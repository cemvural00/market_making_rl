"""
Custom callbacks for stable-baselines3 training.

This module provides early stopping callbacks based on training loss
and evaluation metrics (reward, PnL, etc.).
"""

from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import numpy as np
import os
import tempfile


class RewardBasedEarlyStopping(EvalCallback):
    """
    Early stopping callback based on evaluation reward/performance.
    
    This callback:
    1. Evaluates the agent periodically on an evaluation environment
    2. Monitors a performance metric (default: mean reward per episode)
    3. Saves the best model weights when improvement is detected
    4. Stops training if no improvement for 'patience' evaluations
    5. Automatically restores best weights via EvalCallback mechanism
    
    Parameters
    ----------
    eval_env : gymnasium.Env
        Environment used for evaluation (should be separate from training env)
    monitor : str, default "mean_reward"
        Metric to monitor. Options:
        - "mean_reward": Mean episode reward (default, recommended)
        - "mean_pnl": Mean PnL per episode (extracted from eval env info)
        - "mean_sharpe": Mean Sharpe ratio (if computed from rewards)
    patience : int, default 10
        Number of evaluations without improvement before stopping
    min_delta : float, default 0.0
        Minimum change to qualify as an improvement
    mode : str, default "max"
        "max" for maximizing reward/PnL, "min" for minimizing (not typical)
    best_model_save_path : str or None
        Path to save best model. If None, uses temporary path.
    verbose : int, default 1
        Verbosity level (0 = silent, 1 = print messages)
    eval_freq : int, default 10000
        Evaluate every eval_freq steps
    n_eval_episodes : int, default 10
        Number of episodes to run per evaluation
    deterministic : bool, default True
        Whether to use deterministic actions during evaluation
    render : bool, default False
        Whether to render during evaluation
    """
    
    def __init__(
        self,
        eval_env,
        monitor="mean_reward",
        patience=10,
        min_delta=0.0,
        mode="max",
        best_model_save_path=None,
        verbose=1,
        eval_freq=10000,
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    ):
        # Create temporary save path if not provided
        if best_model_save_path is None:
            temp_dir = tempfile.gettempdir()
            best_model_save_path = os.path.join(temp_dir, "best_model_sb3")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(best_model_save_path) if os.path.dirname(best_model_save_path) else ".", exist_ok=True)
        
        # Wrap eval_env if needed (EvalCallback expects VecEnv)
        # Also wrap with Monitor to track episode statistics properly
        if hasattr(eval_env, 'step') and not hasattr(eval_env, 'num_envs'):  # Single env
            # Store env class and kwargs for proper closure in lambda
            env_class = type(eval_env)
            env_kwargs = {
                'S0': getattr(eval_env, 'S0', 100.0),
                'T': getattr(eval_env, 'T', 1.0),
                'dt': getattr(eval_env, 'dt', 0.0001),
                'A': getattr(eval_env, 'A', 5.0),
                'k': getattr(eval_env, 'k', 1.5),
                'base_delta': getattr(eval_env, 'base_delta', 1.0),
                'max_inventory': getattr(eval_env, 'max_inventory', 20),
                'inv_penalty': getattr(eval_env, 'inv_penalty', 0.01),
            }
            # Wrap with Monitor - create new instance in lambda to avoid closure issues
            eval_env_vec = DummyVecEnv([
                lambda cls=env_class, kwargs=env_kwargs: Monitor(cls(seed=None, **kwargs))
            ])
        else:  # Already a VecEnv
            eval_env_vec = eval_env
        
        super().__init__(
            eval_env=eval_env_vec,
            best_model_save_path=best_model_save_path,
            log_path=None,  # Don't log separately, we handle logging
            eval_freq=eval_freq,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            n_eval_episodes=n_eval_episodes,
        )
        
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.n_eval_episodes = n_eval_episodes
        
        # Track best performance
        self.best_value = -np.inf if mode == "max" else np.inf
        self.wait = 0  # Number of evaluations without improvement
        self.stopped_epoch = 0
        self.best_eval_timestep = 0
        self._should_stop = False  # Flag to signal early stopping
        
        # Store PnL history for computing metrics
        self._pnl_history = []
        self._reward_history = []
        self._last_eval_pnls = []  # Store PnL from last evaluation
        
    def _on_step(self) -> bool:
        """
        Called during training. EvalCallback handles evaluation,
        we override _on_evaluation to check for improvements.
        
        Returns
        -------
        bool
            True to continue training, False to stop
        """
        # Check flag BEFORE calling parent (in case it was set in previous evaluation)
        if self._should_stop:
            if self.verbose > 0:
                print(f"EarlyStopping: Stopping training (flag set) at step {self.num_timesteps}")
            return False
        
        # Call parent to perform evaluation
        # This will call _on_evaluation which updates self.wait and may set _should_stop
        continue_training = super()._on_step()
        
        # If parent stopped evaluation callback, respect that
        if not continue_training:
            return False
        
        # Check flag AFTER evaluation (in case it was just set in _on_evaluation)
        if self._should_stop:
            if self.verbose > 0:
                print(f"EarlyStopping: Stopping training (flag set after evaluation) at step {self.num_timesteps}")
            return False
            
        return True
    
    def _on_evaluation(self) -> None:
        """
        Called after each evaluation. Check if we should stop training.
        Extract PnL if monitoring PnL metric.
        """
        # Get last evaluation results from parent
        if self.last_mean_reward is None:
            return
        
        # Collect reward history
        self._reward_history.append(self.last_mean_reward)
        
        # Try to extract PnL if monitoring PnL
        if self.monitor == "mean_pnl":
            # Run a quick evaluation to extract PnL from info dicts
            # We do this separately to avoid interfering with parent's evaluation
            pnls = []
            try:
                # Get single env from VecEnv if needed
                if hasattr(self.eval_env, 'envs') and len(self.eval_env.envs) > 0:
                    eval_env_single = self.eval_env.envs[0]
                else:
                    eval_env_single = self.eval_env
                
                for _ in range(min(self.n_eval_episodes, 5)):  # Limit to 5 for speed
                    obs, _ = eval_env_single.reset()
                    done = False
                    episode_pnl = None
                    
                    while not done:
                        action, _ = self.model.predict(obs, deterministic=self.deterministic)
                        obs, reward, done, truncated, info = eval_env_single.step(action)
                        done = done or truncated
                        
                        # Extract PnL from info dict if available
                        if isinstance(info, dict) and "pnl" in info:
                            episode_pnl = float(info["pnl"])
                    
                    if episode_pnl is not None:
                        pnls.append(episode_pnl)
            except Exception as e:
                if self.verbose > 0:
                    print(f"Warning: Could not extract PnL directly: {e}")
            
            if pnls:
                current_pnl = np.mean(pnls)
                self._pnl_history.append(current_pnl)
                self._last_eval_pnls = pnls
            else:
                # Fallback: approximate PnL from rewards
                # Since rewards ≈ dPnL - inv_penalty, approximate by accumulating
                if not self._pnl_history:
                    # First evaluation: approximate from reward
                    # Assume mean reward per step * steps per episode
                    steps_per_episode = getattr(eval_env_single, 'n_steps', 10000) if 'eval_env_single' in locals() else 10000
                    current_pnl = self.last_mean_reward * steps_per_episode / self.n_eval_episodes
                else:
                    # Accumulate: approximate PnL change from mean reward
                    steps_per_episode = getattr(eval_env_single, 'n_steps', 10000) if 'eval_env_single' in locals() else 10000
                    pnl_change = self.last_mean_reward * steps_per_episode / self.n_eval_episodes
                    current_pnl = self._pnl_history[-1] + pnl_change
                self._pnl_history.append(current_pnl)
                if self.verbose > 0 and len(self._pnl_history) <= 2:
                    print(f"Note: Approximating PnL from rewards (actual PnL extraction preferred)")
        
        # Determine which metric to use
        if self.monitor == "mean_reward":
            current_value = self.last_mean_reward
        elif self.monitor == "mean_pnl":
            current_value = self._pnl_history[-1] if self._pnl_history else self.last_mean_reward
        elif self.monitor == "mean_sharpe":
            # Compute Sharpe ratio from reward history
            if len(self._reward_history) < 2:
                current_value = 0.0
            else:
                rewards = np.array(self._reward_history[-100:])  # Last 100 evaluations
                if rewards.std() > 1e-10:
                    current_value = rewards.mean() / rewards.std()
                else:
                    current_value = rewards.mean()
        else:
            # Unknown monitor, use reward
            current_value = self.last_mean_reward
            if self.verbose > 0:
                print(f"Warning: Unknown monitor '{self.monitor}', using mean_reward")
        
        # Check for improvement
        if self.mode == "max":
            improved = current_value > (self.best_value + self.min_delta)
        else:
            improved = current_value < (self.best_value - self.min_delta)
        
        # Special case: first evaluation always updates best_value
        # This ensures best_value gets updated from -np.inf to actual value
        if self.best_eval_timestep == 0:
            improved = True
            if self.verbose > 0:
                print(f"EarlyStopping: Evaluation #{len(self._reward_history)} - First evaluation recorded ({self.monitor}={current_value:.6f} at step {self.num_timesteps})")
        
        if improved:
            self.best_value = current_value
            self.wait = 0
            self.best_eval_timestep = self.num_timesteps
            
            if self.verbose > 0:
                print(f"EarlyStopping: Evaluation #{len(self._reward_history)} - {self.monitor} improved to {current_value:.6f} "
                      f"at step {self.num_timesteps} (wait reset to 0)")
        else:
            self.wait += 1
            if self.verbose > 0:
                print(f"EarlyStopping: No improvement for {self.wait}/{self.patience} evaluations "
                      f"(best: {self.best_value:.6f}, current: {current_value:.6f})")
        
        # Check if we should stop - set flag immediately
        if self.wait >= self.patience:
            self._should_stop = True  # Set flag to stop training
            self.stopped_epoch = self.num_timesteps
            if self.verbose > 0:
                print(f"\nEarlyStopping: Patience exceeded! Stopping training at step {self.num_timesteps}")
                print(f"  Best {self.monitor}: {self.best_value:.6f} at step {self.best_eval_timestep}")
                print(f"  Final {self.monitor}: {current_value:.6f}")
                print(f"  Total evaluations: {len(self._reward_history)}")
    
    def _check_save_best_model(self) -> bool:
        """
        Override parent method to check early stopping condition.
        """
        # Call parent to save best model (based on reward)
        parent_result = super()._check_save_best_model()
        
        # Check early stopping condition
        if self.wait >= self.patience:
            return False  # Signal to stop training
        
        return parent_result
    
    def _on_training_end(self) -> None:
        """
        Called when training ends. Best model already saved by parent EvalCallback.
        """
        if self.verbose > 0:
            if self.stopped_epoch > 0:
                print(f"Training stopped early at step {self.stopped_epoch}")
            else:
                print(f"Training completed normally")
            
            # Check if any evaluation actually happened
            if self.last_mean_reward is None:
                print(f"Warning: No evaluations occurred during training.")
                print(f"  This may happen if eval_freq ({self.eval_freq}) is higher than total training steps.")
                print(f"  Best model may not reflect best performance.")
            elif self.best_eval_timestep == 0:
                # Evaluation happened but no improvement was recorded
                # This means the first evaluation was the best (or only one)
                if len(self._reward_history) > 0:
                    # Use the first evaluation as best
                    self.best_value = self._reward_history[0]
                    self.best_eval_timestep = self.num_timesteps  # Use current timestep as approximation
                    print(f"Best {self.monitor}: {self.best_value:.6f} (from first evaluation at step {self.num_timesteps})")
                else:
                    # Fallback to last_mean_reward
                    self.best_value = self.last_mean_reward
                    print(f"Best {self.monitor}: {self.best_value:.6f} (single evaluation)")
            else:
                print(f"Best {self.monitor}: {self.best_value:.6f} at step {self.best_eval_timestep}")
                print(f"  Total evaluations: {len(self._reward_history)}")
        
        # The best model should already be saved by parent EvalCallback
        # The model's best weights are automatically used when best_model_save_path is set
        if self.best_model_save_path is not None and os.path.exists(self.best_model_save_path + ".zip"):
            if self.verbose > 0:
                print(f"Best model saved at: {self.best_model_save_path}.zip")


class EarlyStoppingCallback(BaseCallback):
    """
    Early stopping callback based on training loss.
    
    Monitors the training loss and stops training if no improvement
    is observed for a specified number of episodes.
    
    Parameters
    ----------
    monitor : str, default "loss/policy_loss"
        Which loss to monitor. Options:
        - "loss/policy_loss": Policy loss
        - "loss/value_loss": Value function loss
        - "loss/entropy_loss": Entropy loss
        - "loss/total_loss": Total loss (policy + value + entropy)
    patience : int, default 10
        Number of episodes without improvement before stopping
    n_steps : int, default 1024
        Number of steps per rollout (e.g., 1024 for PPO, 512 for LSTM-PPO)
    episode_length : int
        Number of steps per episode (T / dt, e.g., 10000)
    min_delta : float, default 0.0
        Minimum change to qualify as an improvement
    mode : str, default "min"
        "min" for minimizing loss, "max" for maximizing (not used for loss)
    verbose : int, default 1
        Verbosity level (0 = silent, 1 = print messages)
    """
    
    def __init__(
        self,
        monitor="loss/policy_loss",
        patience=10,
        n_steps=1024,
        episode_length=10000,
        min_delta=0.0,
        mode="min",
        verbose=1
    ):
        super().__init__(verbose)
        self.monitor = monitor
        self.patience = patience  # Now in episodes
        self.n_steps = n_steps
        self.episode_length = episode_length
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        
        self.best_loss = np.inf if mode == "min" else -np.inf
        self.wait = 0  # Wait counter in episodes
        self.stopped_epoch = 0
        self.best_epoch = 0
        self._last_rollout_timesteps = -1
        self._last_episode_count = -1
        self.best_weights_path = None
        
    def _on_step(self) -> bool:
        """
        Called at each training step.
        Only checks loss at rollout boundaries (every n_steps).
        
        Returns
        -------
        bool
            True to continue training, False to stop
        """
        # Only check at rollout boundaries (every n_steps)
        current_rollout = self.num_timesteps // self.n_steps
        
        # Check if we've completed a new rollout
        if current_rollout == self._last_rollout_timesteps:
            return True  # Still in the same rollout, skip check
        
        self._last_rollout_timesteps = current_rollout
        
        # Get current loss from logger
        if self.logger is None:
            return True
            
        # Extract loss value from logger
        # SB3 stores metrics in logger.name_to_value dict
        if not hasattr(self.logger, "name_to_value"):
            return True
        
        # Try to get the monitored loss
        loss = None
        if self.monitor in self.logger.name_to_value:
            loss = self.logger.name_to_value[self.monitor]
        else:
            # Try alternative names
            for key in self.logger.name_to_value.keys():
                if self.monitor.split("/")[-1] in key.lower():
                    loss = self.logger.name_to_value[key]
                    break
        
        if loss is None:
            # Loss not found yet, continue training
            return True
        
        # Calculate how many episodes have passed
        current_episode = self.num_timesteps // self.episode_length
        
        # Check if this is an improvement
        if self.mode == "min":
            improved = loss < (self.best_loss - self.min_delta)
        else:
            improved = loss > (self.best_loss + self.min_delta)
        
        if improved:
            self.best_loss = loss
            self.wait = 0  # Reset episode counter
            self._last_episode_count = current_episode
            self.best_epoch = self.num_timesteps
            
            # Save best model weights
            if self.model is not None:
                if self.best_weights_path is None:
                    temp_dir = tempfile.gettempdir()
                    self.best_weights_path = os.path.join(temp_dir, "best_loss_model_sb3")
                try:
                    self.model.save(self.best_weights_path)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not save best weights: {e}")
            
            if self.verbose > 0:
                print(f"EarlyStopping: Loss improved to {loss:.6f} at episode {current_episode} (step {self.num_timesteps})")
        else:
            # Only increment wait when we've completed a new episode since last check
            if current_episode > self._last_episode_count:
                self.wait += (current_episode - self._last_episode_count)
                self._last_episode_count = current_episode
                if self.verbose > 0 and self.wait % 5 == 0:
                    print(f"EarlyStopping: No improvement for {self.wait} episodes (best: {self.best_loss:.6f})")
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = self.num_timesteps
            if self.verbose > 0:
                print(f"\nEarlyStopping: Stopping training at episode {current_episode} (step {self.num_timesteps})")
                print(f"  Best loss: {self.best_loss:.6f} at step {self.best_epoch}")
                print(f"  Final loss: {loss:.6f}")
            return False  # Stop training
        
        return True  # Continue training
    
    def _on_training_end(self) -> None:
        """Called when training ends. Restore best weights if early stopped."""
        if self.verbose > 0:
            if self.stopped_epoch > 0:
                print(f"Training stopped early at step {self.stopped_epoch}")
            else:
                print(f"Training completed. Best loss: {self.best_loss:.6f} at step {self.best_epoch}")
        
        # Restore best weights if available and training was stopped early
        if self.best_weights_path is not None and os.path.exists(self.best_weights_path + ".zip"):
            if self.model is not None and self.stopped_epoch > 0:
                # Only restore if training was stopped early
                try:
                    # Reload best model
                    from stable_baselines3.common.base_class import BaseAlgorithm
                    self.model = BaseAlgorithm.load(self.best_weights_path, env=self.model.get_env())
                    if self.verbose > 0:
                        print(f"Restored best model weights from step {self.best_epoch}")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not restore best weights: {e}")
        
        # Restore best weights if available and training was stopped early
        if self.best_weights_path is not None and os.path.exists(self.best_weights_path + ".zip"):
            if self.model is not None and self.stopped_epoch > 0:
                # Only restore if training was stopped early
                try:
                    # Reload best model
                    from stable_baselines3.common.base_class import BaseAlgorithm
                    self.model = BaseAlgorithm.load(self.best_weights_path, env=self.model.get_env())
                    if self.verbose > 0:
                        print(f"Restored best model weights from step {self.best_epoch}")
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not restore best weights: {e}")


class LossBasedEarlyStopping(EarlyStoppingCallback):
    """
    Simplified early stopping callback that monitors total loss.
    
    This is a convenience wrapper that defaults to monitoring
    the total training loss.
    
    Parameters
    ----------
    patience : int, default 10
        Number of episodes without improvement before stopping
    n_steps : int, default 1024
        Number of steps per rollout (e.g., 1024 for PPO, 512 for LSTM-PPO)
    episode_length : int
        Number of steps per episode (T / dt, e.g., 10000)
    min_delta : float, default 0.0
        Minimum change to qualify as an improvement
    verbose : int, default 1
        Verbosity level
    """
    
    def __init__(self, patience=10, n_steps=1024, episode_length=10000, min_delta=0.0, verbose=1):
        # Try to find total loss, fall back to policy loss
        super().__init__(
            monitor="train/loss",  # Try total loss first
            patience=patience,
            n_steps=n_steps,
            episode_length=episode_length,
            min_delta=min_delta,
            mode="min",
            verbose=verbose
        )
        
    def _on_step(self) -> bool:
        """
        Override to try multiple loss metrics.
        Only checks loss at rollout boundaries (every n_steps).
        """
        # Only check at rollout boundaries (every n_steps)
        current_rollout = self.num_timesteps // self.n_steps
        
        # Check if we've completed a new rollout
        if current_rollout == self._last_rollout_timesteps:
            return True  # Still in the same rollout, skip check
        
        self._last_rollout_timesteps = current_rollout
        
        if self.logger is None:
            return True
            
        if not hasattr(self.logger, "name_to_value"):
            return True
        
        # Try different loss metrics in order of preference
        loss = None
        loss_keys = [
            "train/loss",           # Total loss
            "train/policy_loss",    # Policy loss
            "train/value_loss",     # Value loss
            "loss/policy_loss",     # Alternative naming
            "loss/value_loss",      # Alternative naming
            "policy_loss",          # Simple naming
            "value_loss",           # Simple naming
        ]
        
        for key in loss_keys:
            if key in self.logger.name_to_value:
                loss = self.logger.name_to_value[key]
                if not hasattr(self, "_monitored_key") or self._monitored_key != key:
                    self._monitored_key = key
                    if self.verbose > 0:
                        print(f"EarlyStopping: Monitoring loss metric: {key}")
                break
        
        # If still not found, search for any loss-like key
        if loss is None:
            for key in self.logger.name_to_value.keys():
                if "loss" in key.lower() and "explained" not in key.lower():
                    loss = self.logger.name_to_value[key]
                    if not hasattr(self, "_monitored_key") or self._monitored_key != key:
                        self._monitored_key = key
                        if self.verbose > 0:
                            print(f"EarlyStopping: Monitoring loss metric: {key}")
                    break
        
        if loss is None:
            return True
        
        # Calculate how many episodes have passed
        current_episode = self.num_timesteps // self.episode_length
        
        # Check if this is an improvement
        improved = loss < (self.best_loss - self.min_delta)
        
        if improved:
            self.best_loss = loss
            self.wait = 0  # Reset episode counter
            self._last_episode_count = current_episode
            self.best_epoch = self.num_timesteps
            
            # Save best model weights
            if self.model is not None:
                if self.best_weights_path is None:
                    temp_dir = tempfile.gettempdir()
                    self.best_weights_path = os.path.join(temp_dir, "best_loss_model_sb3")
                try:
                    self.model.save(self.best_weights_path)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Warning: Could not save best weights: {e}")
            
            if self.verbose > 0:
                print(f"EarlyStopping: Loss improved to {loss:.6f} at episode {current_episode} (step {self.num_timesteps})")
        else:
            # Only increment wait when we've completed a new episode since last check
            if current_episode > self._last_episode_count:
                self.wait += (current_episode - self._last_episode_count)
                self._last_episode_count = current_episode
                if self.verbose > 0 and self.wait % 5 == 0:
                    print(f"EarlyStopping: No improvement for {self.wait} episodes (best: {self.best_loss:.6f})")
        
        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = self.num_timesteps
            if self.verbose > 0:
                print(f"\nEarlyStopping: Stopping training at episode {current_episode} (step {self.num_timesteps})")
                print(f"  Best loss: {self.best_loss:.6f} at step {self.best_epoch}")
                print(f"  Final loss: {loss:.6f}")
            return False  # Stop training
        
        return True  # Continue training
