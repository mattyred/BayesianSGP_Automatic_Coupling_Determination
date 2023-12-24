import torch
import numpy as np
import glob
import os
from pathlib import Path
from itertools import repeat

def set_seed(seed=0):
    """Set seed for reproducibility purpose."""
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader

def get_all_files(pattern):
    """Get all files located in a directory given a pattern."""
    file_paths = glob.glob(pattern)

    if len(file_paths) < 1:
        return None
    else:
        return file_paths
    
def ensure_dir(dirname):
    """Check whether the given directory was created; if not, create a new one.

    Args:
        dirname: string, path to the directory.
    """
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def next_path(path_pattern):
    i = 1
    while os.path.exists(path_pattern % i):
        i = i * 2
    a, b = (i / 2, i)
    while a + 1 < b:
        c = (a + b) / 2 # interval midpoint
        a, b = (c, b) if os.path.exists(path_pattern % c) else (a, c)
    directory = path_pattern % b
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory
