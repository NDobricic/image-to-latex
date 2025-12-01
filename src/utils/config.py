"""Configuration loading and management utilities."""

from pathlib import Path
from typing import Any, Optional

import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load configuration from a YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Handle base config inheritance
    if 'base' in config:
        base_path = Path(config_path).parent / config['base']
        base_config = load_config(base_path)
        config = merge_configs(base_config, config)
        del config['base']
    
    return config


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration (takes precedence)
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def save_config(config: dict, path: Path):
    """
    Save configuration to a YAML file.
    
    Args:
        config: Configuration dictionary
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class Config:
    """Configuration wrapper with attribute access."""
    
    def __init__(self, config_dict: dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)
    
    def to_dict(self) -> dict:
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
    
    @classmethod
    def from_file(cls, path: Path) -> "Config":
        return cls(load_config(path))

