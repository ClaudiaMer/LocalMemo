import torch
import numpy as np
from torch.nn.parameter import Parameter
from local_memorization.models.u_net import UNet

class TinyModel(torch.nn.Module):

    def __init__(self, dim, width=200):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(dim, width)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(width, dim)
        self.softmax = torch.nn.Softmax()

    def forward(self, x, t):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x
    

    
class AffineLinear(torch.nn.Module):
    def __init__(self, dim, sqrt_alpha_bars, device, init_scaling="unicircle",
                  init_sigma_b=None, fullweighttensor=True):
        super(AffineLinear, self).__init__()
        # initialize such that all eigvals lie in unit circle 
        # (https://en.wikipedia.org/wiki/Circular_law)
        self.dim = dim
        self.device = device
        self.sqrt_alpha_bars = sqrt_alpha_bars.to(self.device)
        self.T = len(sqrt_alpha_bars)
        self.fullweighttensor = fullweighttensor
        # set init variances
        if init_scaling == "unicircle":
            init_scaling = 1.0/ np.sqrt(dim)
        self.init_scaling = init_scaling
        if init_sigma_b is None: 
            self.init_sigma_b = self.init_scaling
        else: 
            self.init_sigma_b = init_sigma_b
        # init weights
        if self.fullweighttensor:
            self._weights = Parameter(torch.randn(self.T,\
                                                self.dim,
                                                self.dim).to(self.device)
                                     *self.init_scaling)
            self.get_weight = None
        else: 
            self.get_weight = None
        
        self.biases = Parameter(torch.randn(self.T,
                                             self.dim).to(self.device)
                                *self.init_sigma_b)
        if self.init_sigma_b == 0: 
            self.biases.requires_grad = False

    def weights(self,t): 
        if self.fullweighttensor: 
            return self._weights[t]
        else:
            return self.get_weight(t)

    def forward(self, x, t):
        if type(t) == int:
            x_ = x - self.sqrt_alpha_bars[t] * self.biases[t]
            out = torch.einsum("tij, nj -> tni", self.weights(t), x_)
        elif type(t)== torch.Tensor: 
            if t.dim()==0:
                x_ = x - self.sqrt_alpha_bars[t] * self.biases[t]
                out = torch.einsum("tij, nj -> tni", self.weights(t.item()), x_)
            else:
                x_ = x - torch.einsum("t, ti -> ti", self.sqrt_alpha_bars[t], self.biases[t])
                out = torch.einsum("tij, tj -> ti", self.weights(t), x_)
        return out
    
class TwoLayer(torch.nn.Module): 
    """One independent 2-layer NN per diffusion step"""
    def __init__(self, dim, sqrt_alpha_bars, device, init_scaling="unicircle",
                  init_sigma_b=None, fullweighttensor=True):
        super(TwoLayer, self).__init__()
        self.affine_linear_1 = AffineLinear(dim, sqrt_alpha_bars, device,
                                            init_scaling=init_scaling,
                                            init_sigma_b=init_sigma_b,
                                            fullweighttensor=fullweighttensor)
        self.activation = torch.nn.ReLU()
        self.affine_linear_2 = AffineLinear(dim, sqrt_alpha_bars, device,
                                            init_scaling=init_scaling,
                                            init_sigma_b=init_sigma_b,
                                            fullweighttensor=fullweighttensor)
        
    def forward(self, x, t): 
        x = self.affine_linear_1(x,t)
        x = self.activation(x)
        x = self.affine_linear_2(x,t)
        return x
    
class TwoLayerTanh(torch.nn.Module): 
    """One independent 2-layer NN per diffusion step"""
    def __init__(self, dim, sqrt_alpha_bars, device, init_scaling="unicircle",
                  init_sigma_b=None, fullweighttensor=True):
        super(TwoLayerTanh, self).__init__()
        self.affine_linear_1 = AffineLinear(dim, sqrt_alpha_bars, device,
                                            init_scaling=init_scaling,
                                            init_sigma_b=init_sigma_b,
                                            fullweighttensor=fullweighttensor)
        self.activation = torch.nn.Tanh()
        self.affine_linear_2 = AffineLinear(dim, sqrt_alpha_bars, device,
                                            init_scaling=init_scaling,
                                            init_sigma_b=init_sigma_b,
                                            fullweighttensor=fullweighttensor)
        
    def forward(self, x, t): 
        x = self.affine_linear_1(x,t)
        x = self.activation(x)
        x = self.affine_linear_2(x,t)
        return x

class ReluAffineLinear(torch.nn.Module):
    def __init__(self, dim, sqrt_alpha_bars, device, init_scaling="unicircle",
                  init_sigma_b=None, fullweighttensor=True):
        super(ReluAffineLinear, self).__init__()

        # initialize such that all eigvals lie in unit circle 
        # (https://en.wikipedia.org/wiki/Circular_law)

        self.dim = dim
        self.device = device
        self.sqrt_alpha_bars = sqrt_alpha_bars.to(self.device)
        self.T = len(sqrt_alpha_bars)

        self.fullweighttensor = fullweighttensor
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(dim, dim)

        # set init variances
        if init_scaling == "unicircle":
            init_scaling = 1.0/ np.sqrt(dim)
        self.init_scaling = init_scaling
        
        if init_sigma_b is None: 
            self.init_sigma_b = self.init_scaling
        else: 
            self.init_sigma_b = init_sigma_b
        # init weights
        if self.fullweighttensor:
            self._weights = Parameter(torch.randn(self.T,\
                                                self.dim,
                                                self.dim).to(self.device)
                                     *self.init_scaling)
            self.get_weight = None
        else: 
            self.get_weight = None
        
        self.biases = Parameter(torch.randn(self.T,
                                             self.dim).to(self.device)
                                *self.init_sigma_b)
        if self.init_sigma_b == 0: 
            self.biases.requires_grad = False

    def weights(self,t): 
        if self.fullweighttensor: 
            return self._weights[t]
        else:
            return self.get_weight(t)

    def forward(self, x, t):
        if type(t) == int:
            x_ = x - self.sqrt_alpha_bars[t] * self.biases[t]
            out = torch.einsum("tij, nj -> tni", self.weights(t), x_)
        elif type(t)== torch.Tensor: 
            if t.dim()==0:
                x_ = x - self.sqrt_alpha_bars[t] * self.biases[t]
                out = torch.einsum("tij, nj -> tni", self.weights(t.item()), x_)
            else:
                x_ = x - torch.einsum("t, ti -> ti", self.sqrt_alpha_bars[t], self.biases[t])
                out = torch.einsum("tij, tj -> ti", self.weights(t), x_)
        return self.linear2(self.activation(out))
    

