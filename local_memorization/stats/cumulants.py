import torch
import numpy as np


def mean(dataset): 
    return torch.mean(dataset, dim=0)

def cov(dataset): 
    return torch.cov(dataset.T)

def center(dataset):
    """Center a dataset

    Args:
        dataset (torch.Tensor): data, organized as (sample_id, dim1, dim2, ...)

    Returns:
        torch.Tensor: centered data
    """
    orig_shape = tuple(dataset.shape)
    flat_shape = (orig_shape[0],np.prod(orig_shape[1:]))
    print(flat_shape)
    mean_flattened_shape = mean(dataset.reshape(flat_shape))
    mean_ = mean_flattened_shape.reshape(orig_shape[1:])
    return dataset - mean_

def cov_decompositon(cov_):
    """Compute eigendecomposition of covariance matrix

    Args:
        cov_ (torch.Tensor): covariance matrix

    Returns:
        eigvals, eigvecs: eigenvalues and eigenvectors of covariance, s.t. 
            cov = eigvecs @ (torch.diag(eigvals) @ eigvecs.T)
    """ 
    eigvals, eigvecs = torch.linalg.eigh(cov_)
    return eigvals, eigvecs

def cov_decompositon_from_data(dataset):
    """Compute eigendecomposition of empirical covariance matrix of data

    Args:
        dataset (torch.Tensor): Dataset organized as (sample_index, data_dim)

    Returns:
        eigvals, eigvecs: eigenvalues and eigenvectors of covariance, s.t. 
            cov = eigvecs @ (torch.diag(eigvals) @ eigvecs.T)
    """ 
    cov_ = cov(dataset)
    return cov_decompositon(cov_)

