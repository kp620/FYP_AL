import torch

def check_device():
    USE_CPU  = True
    dtype = torch.float64
    if USE_CPU and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    print(device)
    return device, dtype

