# utils/reproducibility.py
"""Reproducibility helpers used by training/evaluation scripts."""

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set Python/NumPy/PyTorch seeds.

    deterministic=True improves reproducibility at the cost of possible speed loss.
    It is safe for paper experiments and smoke tests.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # Do not force torch.use_deterministic_algorithms(True): some CUDA ops used by
        # baselines may not have deterministic kernels on every server.
    else:
        # Restore the fast cuDNN autotuner path. Seeds are still fixed, but exact
        # bitwise reproducibility is not guaranteed when non-deterministic kernels are used.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function compatible with torch.Generator seeds."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
