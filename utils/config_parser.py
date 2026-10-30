# utils/config_parser.py
import yaml

class Config:
    def __init__(self, dict_config):
        for key, value in dict_config.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)

def load_config(config_path='config.yaml'):
    """
    Loads the YAML configuration file without intercepting command-line arguments.
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
        
    return Config(config_dict)