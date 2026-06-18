import torch 
import numpy as np
torch.set_default_dtype(torch.float32)

class Gaussian:
    def __init__(self, dim, device, mean=None, covariance=None): 
        self.dim = dim 
        self.device = device
        if mean is None: 
            mean = torch.zeros(dim).float()
        if covariance is None: 
            covariance = torch.eye(dim)

        if type(covariance) == torch.Tensor:
            self.covariance = covariance.to(device)
        else: 
            self.covariance = covariance
        if type(mean) == torch.Tensor:
            self.mean = mean.to(device)
        else: 
            self.mean = mean

        self.eigvals, self.eigvecs = \
            torch.linalg.eigh(self.covariance)
        
        self.eigvals = abs(self.eigvals) # avoids negative eigenvalues due 
        #to finite numerical precision

        self.sample_matrix = \
            self.eigvecs @ torch.diag(torch.sqrt(self.eigvals))
        self.pca_matrix = \
            torch.diag(1.0/torch.sqrt(self.eigvals)) @ self.eigvecs.T
        
    def sample(self, num_samples, center=False, latents=None): 
        if latents is None:
            samples_white = torch.randn(num_samples, self.dim).to(self.device)
        else: 
            samples_white = latents
        samples = (self.sample_matrix @ samples_white.T ).T \
            + self.mean
        if center:
            mean_emp = torch.mean(samples, dim=0)
            assert mean_emp.shape == (self.dim,)
            self.mean_for_emp_center = mean_emp
            samples -= mean_emp
        self.samples = samples
        return self.samples
    
    def whiten(self, samples): 
        samples = samples.to(self.device)
        return (self.pca_matrix @ (samples-self.mean).T ).T
    
    @property
    def covariance_data(self,): 
        return torch.cov(self.samples.T)
    
    @property
    def mean_data(self,): 
        return torch.mean(self.samples, dim=0)
    
    @property
    def proj_covariance_data_eigspace(self,): 
        self.eigvals_data, self.eigvecs_data = \
            torch.linalg.eigh(self.covariance_data)
        return self.eigvecs_data