import os
import torch
import numpy as np
import random


def set_seed(seed_value=42):
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Raise error on non-deterministic ops
    torch.use_deterministic_algorithms(True)

    print(f"[INFO] Seed set to {seed_value}")


def deterministic_randperm(length: int, seed: int = 42):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randperm(n=length, generator=g)
