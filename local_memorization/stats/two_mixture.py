import numpy as np
import torch

def sample(N, mu=1.0, sigma=0.5): 

    rands = torch.randn(N)*sigma 

    binary = torch.sign(torch.randn(N))*mu # either my or -mu
    return rands + binary