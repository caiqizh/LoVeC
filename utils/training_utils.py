import re
import os, sys, json, time
import importlib
import torch
import torch.nn as nn
from config import TrainingConfig

def load_config(config_path=None):
    """Load configuration from file or use default"""
    if config_path and os.path.exists(config_path):
        # Get the directory containing the config file
        config_dir = os.path.dirname(os.path.abspath(config_path))
        # Get the filename without extension
        config_name = os.path.basename(config_path).replace(".py", "")
        
        # Add the directory to Python path if it's not already there
        if config_dir not in sys.path:
            sys.path.insert(0, config_dir)
            
        try:
            # Import the module
            config_module = importlib.import_module(config_name)
            # Get the cfg variable from the module
            cfg = getattr(config_module, "cfg", None)
            if cfg is None:
                print(f"Warning: No 'cfg' variable found in {config_path}, using default config")
                cfg = TrainingConfig()
        except ImportError as e:
            print(f"Error importing config: {e}")
            print("Using default config")
            cfg = TrainingConfig()
    else:
        cfg = TrainingConfig()
    return cfg

def save_config(cfg, output_dir):
    """Save configuration to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    config_dict = {k: v if not isinstance(v, list) or not v or not isinstance(v[0], type) else [item.__name__ for item in v] 
                  for k, v in cfg.__dict__.items()}
    
    with open(os.path.join(output_dir, f"config_{timestamp}.json"), "w") as f:
        json.dump(config_dict, f, indent=2)




