import torch
from Gaussian import Gaussian
device = torch.device("cuda")
#torch.set_default_dtype(torch.float64)


dim = 3
covariance = torch.tensor([[2, 1, 0.2],
                           [1, 2, -0.1],
                           [0.2, -0.1, 3]]).to(device)
mean = torch.tensor([1,4.0, -2.0]).to(device)

g = Gaussian(dim=dim, device=device, mean=mean, covariance=covariance)

for a,b in zip([g.mean, g.covariance], [mean, covariance]): 
    assert torch.allclose(a,b)

samples = g.sample(dim*10000)
assert samples.shape == (dim*10000, dim)

for a,b in zip([g.mean_data, g.covariance_data], [mean, covariance]): 
    assert torch.allclose(a,b, atol=1e-1)

samples_white = g.whiten(samples)
samples_white_mean = torch.mean(samples_white, dim=0)
samples_white_cov = torch.cov(samples_white.T)

for a,b in zip([samples_white_mean, samples_white_cov],
               [torch.zeros(dim).to(device), torch.eye(dim).to(device)]): 
    assert torch.allclose(a,b, atol=2e-2)

g1 = Gaussian(dim=dim, device=device, mean=mean, covariance=covariance)
samples1 = g1.sample(2)
proj_matr = g1.proj_covariance_data_eigspace
assert torch.allclose(proj_matr @ proj_matr.T, torch.eye(dim).to(g1.device))
assert torch.allclose(proj_matr.T @ ( g1.covariance_data @ proj_matr), torch.diag(g1.eigvals_data))

print("Gauss tests successful")