"""
Optuna search space definitions for each RL agent type.

Each sampler function takes an optuna.Trial and returns a config dict
that can be passed directly to the corresponding agent's __init__.

Architecture choices are stored as categorical string keys so Optuna can
log them; the actual list values are reconstructed by reconstruct_policy_kwargs()
when exporting results.
"""

# Candidate values ─────────────────────────────────────────────────────────────

# All batch sizes divide all n_steps choices, so they can be sampled independently.
_PPO_N_STEPS    = [512, 1024, 2048]
_PPO_BATCH      = [64, 128, 256, 512]
_LSTM_N_STEPS   = [256, 512, 1024]
_LSTM_BATCH     = [64, 128, 256]
_OFF_POLICY_BUF = [50_000, 100_000, 200_000]
_OFF_POLICY_BAT = [64, 128, 256, 512]

# Named architecture options (string key → list) used for categorical sampling.
NET_ARCH_OPTIONS = {
    "small":        [128, 128],
    "medium":       [256, 256],
    "medium_deep":  [256, 256, 128],
    "large":        [256, 256, 256],
}

LSTM_NET_ARCH_OPTIONS = {
    "tiny":   [32, 32],
    "small":  [64, 64],
    "medium": [128, 128],
}


# Sampler functions ─────────────────────────────────────────────────────────────

def sample_ppo(trial):
    """Search space for PPOAgent (standard MLP policy)."""
    return {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps":       trial.suggest_categorical("n_steps", _PPO_N_STEPS),
        "batch_size":    trial.suggest_categorical("batch_size", _PPO_BATCH),
        "n_epochs":      trial.suggest_int("n_epochs", 2, 10),
        "gamma":         trial.suggest_float("gamma", 0.9, 1.0),
        "ent_coef":      trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
        "clip_range":    trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "gae_lambda":    trial.suggest_float("gae_lambda", 0.8, 1.0),
        "vf_coef":       0.5,
        "verbose":       0,
        "use_vec_env":   False,
    }


def sample_deep_ppo(trial):
    """Search space for DeepPPOAgent (PPO with deeper MLP)."""
    params = sample_ppo(trial)
    arch_key = trial.suggest_categorical("net_arch", list(NET_ARCH_OPTIONS.keys()))
    params["policy_kwargs"] = {
        "net_arch":      NET_ARCH_OPTIONS[arch_key],
        "activation_fn": "ReLU",
    }
    return params


def sample_lstm_ppo(trial):
    """Search space for LSTMPPOAgent (RecurrentPPO with LSTM)."""
    arch_key = trial.suggest_categorical("lstm_net_arch", list(LSTM_NET_ARCH_OPTIONS.keys()))
    return {
        "learning_rate":  trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "n_steps":        trial.suggest_categorical("n_steps", _LSTM_N_STEPS),
        "batch_size":     trial.suggest_categorical("batch_size", _LSTM_BATCH),
        "n_epochs":       trial.suggest_int("n_epochs", 2, 10),
        "gamma":          trial.suggest_float("gamma", 0.9, 1.0),
        "ent_coef":       trial.suggest_float("ent_coef", 1e-5, 0.05, log=True),
        "clip_range":     trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3]),
        "gae_lambda":     trial.suggest_float("gae_lambda", 0.8, 1.0),
        "vf_coef":        0.5,
        "verbose":        0,
        "use_vec_env":    False,
        "policy_kwargs": {
            "lstm_hidden_size": trial.suggest_categorical("lstm_hidden_size", [32, 64, 128, 256]),
            "n_lstm_layers":    trial.suggest_int("n_lstm_layers", 1, 2),
            "shared_lstm":      False,
            "net_arch":         LSTM_NET_ARCH_OPTIONS[arch_key],
            "activation_fn":    "ReLU",
        },
    }


def sample_sac(trial):
    """Search space for SACAgent.

    gradient_steps is fixed at 1 (not tuned). With train_freq in [1,4] and
    n_train_steps_offpolicy=10k, worst-case gradient updates = 10k — vs the
    original 120k when gradient_steps was tuned up to 4 at 30k steps.
    Fixing it at 1 is the SB3 default and sufficient to rank configs reliably.
    """
    return {
        "learning_rate":   trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "buffer_size":     trial.suggest_categorical("buffer_size", _OFF_POLICY_BUF),
        "batch_size":      trial.suggest_categorical("batch_size", _OFF_POLICY_BAT),
        "tau":             trial.suggest_float("tau", 0.001, 0.05, log=True),
        "gamma":           trial.suggest_float("gamma", 0.9, 1.0),
        "train_freq":      trial.suggest_int("train_freq", 1, 4),
        "gradient_steps":  1,
        "learning_starts": 100,
        "ent_coef":        "auto",
        "verbose":         0,
        "use_vec_env":     False,
    }


def sample_lstm_sac(trial):
    """Search space for LSTMSACAgent (deep SAC with custom net_arch)."""
    params = sample_sac(trial)
    arch_key = trial.suggest_categorical("net_arch", list(NET_ARCH_OPTIONS.keys()))
    params["policy_kwargs"] = {
        "net_arch":      NET_ARCH_OPTIONS[arch_key],
        "activation_fn": "ReLU",
    }
    return params


def sample_td3(trial):
    """Search space for TD3Agent.

    gradient_steps is fixed at 1 for the same reason as SACAgent — see
    sample_sac docstring. TD3's twin-critic structure already doubles the
    gradient work vs a single-critic agent, so fixing gradient_steps=1
    avoids a further 4× blowup.
    """
    return {
        "learning_rate":       trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True),
        "buffer_size":         trial.suggest_categorical("buffer_size", _OFF_POLICY_BUF),
        "batch_size":          trial.suggest_categorical("batch_size", _OFF_POLICY_BAT),
        "tau":                 trial.suggest_float("tau", 0.001, 0.05, log=True),
        "gamma":               trial.suggest_float("gamma", 0.9, 1.0),
        "train_freq":          trial.suggest_int("train_freq", 1, 4),
        "gradient_steps":      1,
        "policy_delay":        trial.suggest_int("policy_delay", 1, 4),
        "target_policy_noise": trial.suggest_float("target_policy_noise", 0.1, 0.4),
        "target_noise_clip":   trial.suggest_float("target_noise_clip", 0.3, 0.7),
        "learning_starts":     100,
        "verbose":             0,
        "use_vec_env":         False,
    }


# Registry ─────────────────────────────────────────────────────────────────────

SEARCH_SPACE_REGISTRY = {
    "PPOAgent":     sample_ppo,
    "DeepPPOAgent": sample_deep_ppo,
    "LSTMPPOAgent": sample_lstm_ppo,
    "SACAgent":     sample_sac,
    "LSTMSACAgent": sample_lstm_sac,
    "TD3Agent":     sample_td3,
}


def get_search_space_fn(agent_name):
    """Return the sampler function for the given agent class name."""
    if agent_name not in SEARCH_SPACE_REGISTRY:
        raise ValueError(
            f"No search space defined for agent: {agent_name}. "
            f"Available: {list(SEARCH_SPACE_REGISTRY.keys())}"
        )
    return SEARCH_SPACE_REGISTRY[agent_name]


def reconstruct_policy_kwargs(agent_name, trial_params):
    """
    Rebuild the policy_kwargs dict from raw Optuna trial params.

    Called during export: Optuna stores primitive values (strings, ints, floats)
    for categorical choices. This function converts those back to the nested
    policy_kwargs structure expected by agent __init__.

    Parameters
    ----------
    agent_name : str
        Agent class name.
    trial_params : dict
        Mutable copy of trial.params (will be modified in-place).

    Returns
    -------
    dict or None
        policy_kwargs dict, or None if the agent has no policy_kwargs.
    """
    if agent_name == "DeepPPOAgent":
        arch_key = trial_params.pop("net_arch", "medium")
        return {
            "net_arch":      NET_ARCH_OPTIONS.get(arch_key, [256, 256]),
            "activation_fn": "ReLU",
        }

    if agent_name == "LSTMSACAgent":
        arch_key = trial_params.pop("net_arch", "medium")
        return {
            "net_arch":      NET_ARCH_OPTIONS.get(arch_key, [256, 256]),
            "activation_fn": "ReLU",
        }

    if agent_name == "LSTMPPOAgent":
        arch_key           = trial_params.pop("lstm_net_arch", "small")
        lstm_hidden_size   = trial_params.pop("lstm_hidden_size", 128)
        n_lstm_layers      = trial_params.pop("n_lstm_layers", 1)
        return {
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers":    n_lstm_layers,
            "shared_lstm":      False,
            "net_arch":         LSTM_NET_ARCH_OPTIONS.get(arch_key, [64, 64]),
            "activation_fn":    "ReLU",
        }

    return None
