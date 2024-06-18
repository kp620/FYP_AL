"""
Check the device and data type
"""

import torch

def check_device():
    USE_CPU  = True
    dtype = torch.float
    if USE_CPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)
    return device, dtype

