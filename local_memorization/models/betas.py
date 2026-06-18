import torch 
import numpy as np


def linear(mi=1e-4, ma=2e-2, T=1000): 
    if T ==1000:
        ts = torch.arange(T)
        betas = (ma-mi)/(T-1)*ts + mi
        torch.linspace(ma, mi, T, dtype=torch.float32)
    elif T == 10:
        ts = torch.arange(T)/T
        betas = ts
    else: 
        betas = 0.01*torch.ones(T)
    return betas

def linear_scaled(mi=1e-4, ma=2e-2, T=1000): 
    ma = min(0.9,ma*1000/T)
    mi = max(mi, mi*1000/T)
    intermediate = 1.0/T
    betas = torch.zeros(T)
    betas[:T//2] = torch.linspace(mi, intermediate, T//2)
    betas[T//2:] = torch.linspace(intermediate, ma, T//2)
    #ts = torch.arange(T)
    #betas = (ma-mi)/(T-1)*ts + mi
    #torch.linspace(ma, mi, T, dtype=torch.float32)
    return betas