import random
import numpy as np
import yaml


def load_config(path: str):
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass  # src/data.py does not need PyTorch