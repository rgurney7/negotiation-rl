"""Seed local RNGs. Buyer is a remote API model; residual buyer variance is nondeterministic, so report over multiple seeds."""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # No torch.use_deterministic_algorithms(True): some fused LLM kernels lack deterministic impls and would raise.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
