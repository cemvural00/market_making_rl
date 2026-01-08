"""
Utilities for loading trained RL agent models.
"""

import os
import json
from typing import Optional


def load_trained_agent(agent_class, env, model_path: str, agent_config: Optional[dict] = None):
    """
    Load a trained RL agent from disk.

    Parameters
    ----------
    agent_class : class
        Agent class (e.g., PPOAgent, DeepPPOAgent, LSTMPPOAgent)
    env : gymnasium.Env
        Environment instance (required for loading SB3 models)
    model_path : str
        Path to saved model (without .zip extension)
    agent_config : dict, optional
        Agent configuration dict (used if model metadata not found)

    Returns
    -------
    agent
        Loaded and ready-to-use agent instance
    """
    # Check if model exists
    model_file = f"{model_path}.zip"
    if not os.path.exists(model_file):
        raise FileNotFoundError(f"Model file not found: {model_file}")

    # Try to load metadata
    metadata_path = os.path.join(os.path.dirname(model_path), "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        # Use config from metadata if agent_config not provided
        if agent_config is None:
            agent_config = metadata.get("agent_config", {})
    elif agent_config is None:
        agent_config = {}

    # Instantiate agent
    try:
        agent = agent_class(env, config=agent_config)
    except TypeError:
        # Some agents might take only config
        agent = agent_class(config=agent_config)

    # Load the trained model
    agent.load(model_path, env=env)

    return agent


def get_model_path(env_name: str, agent_name: str, model_save_path: str = "models") -> str:
    """
    Get the expected model path for a given environment and agent.

    Parameters
    ----------
    env_name : str
        Environment class name (e.g., "ABMVanillaEnv")
    agent_name : str
        Agent class name (e.g., "PPOAgent")
    model_save_path : str
        Base directory for models

    Returns
    -------
    str
        Path to model (without .zip extension)
    """
    return os.path.join(model_save_path, env_name, agent_name, "model")


def model_exists(env_name: str, agent_name: str, model_save_path: str = "models") -> bool:
    """
    Check if a trained model exists for the given environment and agent.

    Parameters
    ----------
    env_name : str
        Environment class name
    agent_name : str
        Agent class name
    model_save_path : str
        Base directory for models

    Returns
    -------
    bool
        True if model exists, False otherwise
    """
    model_path = get_model_path(env_name, agent_name, model_save_path)
    model_file = f"{model_path}.zip"
    return os.path.exists(model_file)
