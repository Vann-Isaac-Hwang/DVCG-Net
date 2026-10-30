# utils/config_parser.py
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


class Config:
    def __init__(self, dict_config):
        for key, value in dict_config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(config_path=None):
    """
    Loads the YAML configuration file without intercepting command-line arguments.
    """
    path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open('r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    return Config(config_dict)
