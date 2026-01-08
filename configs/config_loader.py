import yaml
import os
from typing import Union, Dict, Any


def load_yaml_config(path: str) -> Dict[str, Any]:
    """
    Loads a YAML configuration file and returns it as a Python dict.

    Supports relative paths inside the project structure.

    Parameters
    ----------
    path : str
        Path to YAML file (relative or absolute).

    Returns
    -------
    dict
        Configuration dictionary loaded from YAML.

    Examples
    --------
    >>> config = load_yaml_config("configs/hyperparams/abm_env.yaml")
    >>> config = load_yaml_config("configs/env_configs.yaml")
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_config(config: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Flexible config loader: accepts either a YAML file path or a dict.

    This allows seamless switching between YAML files and Python dicts
    without changing your code.

    Parameters
    ----------
    config : str or dict
        Either a path to a YAML file, or a dictionary of config values.

    Returns
    -------
    dict
        Configuration dictionary.

    Examples
    --------
    >>> # From YAML file
    >>> env_config = load_config("configs/env_configs.yaml")
    >>> env_config = load_config("configs/hyperparams/abm_env.yaml")
    >>>
    >>> # From nested YAML (access specific key)
    >>> all_configs = load_config("configs/env_configs.yaml")
    >>> abm_config = all_configs["abm_vanilla"]
    >>>
    >>> # From dict (passes through unchanged)
    >>> env_config = load_config({"S0": 100, "sigma": 2.0})
    """
    if isinstance(config, dict):
        return config
    elif isinstance(config, str):
        return load_yaml_config(config)
    else:
        raise TypeError(
            f"Config must be dict or str (file path), got {type(config)}"
        )
