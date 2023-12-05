import os
import argparse
import time
import json
import torch
import numpy as np

from src.misc.settings import settings

device = settings.device
if device.type == 'cuda':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

if __name__ == '__main__':
    pass