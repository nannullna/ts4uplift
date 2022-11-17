import os
import json
import random
import numpy as np
import torch

def set_seed(seed: int):
    """Set all random number generators' seed values for reproducibility.
    
    Args:
        seed: int
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def write_json(obj, path: str, verbose: bool=False, **kwargs) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, **kwargs)
    if verbose:
        print(f"Results saved to {path}.")