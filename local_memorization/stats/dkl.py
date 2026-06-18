import torch 
import numpy as np
from local_memorization.stats.Gaussian import Gaussian
from scipy.linalg import eigh

def stablelogdet(cov, eigvals=None):

    cov = (cov + cov.T)/2
    if eigvals is None:
        eigvals = eigh(cov, eigvals_only=True)
    assert np.all(eigvals >=0)

    #assert torch.all(eigs.imag ==0)

    logdet = sum(np.log(eigvals))
    return logdet

def DKL(gauss1, gauss2, eigvals1=None, eigvals2=None, inv2=None): 

    if eigvals1 is None:
        eigvals = eigh(gauss1.covariance, eigvals_only=True)
    cov1 = np.array(gauss1.covariance)
    cov2 = np.array(gauss2.covariance)
    logdet1 = stablelogdet(cov1, eigvals=eigvals1)#torch.linalg.det(gauss1.covariance)
    logdet2 = stablelogdet(cov2, eigvals=eigvals2)# torch.linalg.det(gauss2.covariance)
    if inv2 is None:
        inv2 = torch.linalg.inv(cov2)
    mean_diff = gauss1.mean - gauss2.mean
    dkl = 0.5*(+logdet2 - logdet1 - gauss1.dim
               + mean_diff.T @ inv2 @ mean_diff
               + np.trace(inv2 @ cov1))
    return dkl


if __name__ == "__main__": 
    dim = 10
    gauss1 = Gaussian(dim, torch.device("cpu"), 
                      covariance=torch.diag(torch.randn(dim)**2), 
                      mean = torch.randn(dim))
    gauss2 = Gaussian(dim, gauss1.device,
                      covariance=torch.clone(gauss1.covariance), 
                      mean=torch.clone(gauss1.mean))
    assert DKL(gauss1, gauss2).item()==0


