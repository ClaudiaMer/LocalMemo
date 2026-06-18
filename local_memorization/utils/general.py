
import torch
import numpy as np

def steps2epochs(steps,batch_size,dataset_size): 
    steps_in_epoch = dataset_size//batch_size
    n_epochs = steps//steps_in_epoch
    return n_epochs


def copy_data(N_out, N, train_data): 
    # repeat data such that there are at least N_out data entries
    if N >= N_out: 
        return train_data[:N_out]
    else: 
        num_times = int(np.ceil(N_out/N))
        stacked_data = torch.vstack([train_data,]*num_times)
        return stacked_data[:N_out]
