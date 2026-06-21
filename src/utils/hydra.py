"""
Hydra configuration management utilities.

This module provides helpers for loading Hydra and YAML configurations,
and preparing experiments by saving the resolved configuration.
"""

import os
from typing import Any, Dict, List, Optional

import yaml
from hydra.compose import compose
from hydra.initialize import initialize
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf

from .logger import create_logger

logger = create_logger(__name__)


def load_hydra_config(
    hydra_config: DictConfig,
) -> Dict[str, Any]:
    """
    Load hydra config and returns ready-to-use dict.

    Notes:
        This function also restores current working directory (Hydra change it internally)

    Args:
        hydra_config:

    Returns:

    """
    os.chdir(get_original_cwd())
    return yaml.load(
        OmegaConf.to_yaml(hydra_config, resolve=True), Loader=yaml.FullLoader
    )


def load_hydra_config_from_path(
    config_path: str,
    config_name: str,
    overrides: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load a Hydra configuration from a specific file path.

    Args:
        config_path (str): Path to the configuration directory.
        config_name (str): Name of the configuration file (without extension).
        overrides (Optional[List[str]]): Hydra overrides list.

    Returns:
        Dict[str, Any]: The resolved configuration as a dictionary.
    """
    if overrides is None:
        overrides = list()
    rel_module_path = os.path.relpath(os.getcwd(), os.path.dirname(__file__))
    with initialize(config_path=os.path.join(rel_module_path, config_path)):
        return OmegaConf.to_container(
            compose(config_name=config_name, overrides=overrides), resolve=True
        )


def load_yaml(
    yaml_path: str,
) -> Dict[str, Any]:
    """
    Load a YAML file.

    Args:
        yaml_path (str): Path to the YAML file.

    Returns:
        Dict[str, Any]: Loaded dictionary.
    """
    with open(yaml_path, "r") as f:
        return yaml.safe_load(f)


def prepare_experiment(
    hydra_config: DictConfig,
) -> Dict[str, Any]:
    """
    Prepare the experiment directory and save the configuration.

    Args:
        hydra_config (DictConfig): The Hydra configuration object.

    Returns:
        Dict[str, Any]: The resolved configuration.
    """
    experiment_dir = os.getcwd()
    save_path = os.path.join(experiment_dir, "experiment_config.yaml")
    OmegaConf.set_struct(hydra_config, False)
    hydra_config["yaml_path"] = save_path
    hydra_config["experiment"]["folder"] = experiment_dir
    logger.info(OmegaConf.to_yaml(hydra_config, resolve=True))
    config = load_hydra_config(hydra_config)
    with open(save_path, "w") as f:
        OmegaConf.save(config=config, f=f.name)
    return config
