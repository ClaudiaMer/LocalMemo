import torch 
import numpy as np

def track_loss(prev_tracks, diffmodel, step, x_0, loss=None, **args):
    """track theo loss prediction during training

    Args:
        prev_tracks (torch.Tensor or None): previously tracked eigenvalues
        diffmodel (Diffusion1d): Diffusion Model
        step (int): training step number
        x_0 (torch.Tensor): training Data
        loss: loss of network
    Returns:
        torch.Tensor: Losses per training step
    """
    if prev_tracks == None: 
        losses = loss.unsqueeze(0).detach()
    else: 
        losses = torch.cat((prev_tracks,  loss.unsqueeze(0).detach()), 0)
    return losses

def get_test_loss_track_fn(test_data, t = None):
    def track_test_loss(prev_tracks, diffmodel, step, x_0, loss=None, **args):
        """track test loss during training

        Args:
            prev_tracks (torch.Tensor or None): previously tracked eigenvalues
            diffmodel (Diffusion1d): Diffusion Model
            step (int): training step number
            x_0 (torch.Tensor): training Data
            loss: loss of network
        Returns:
            torch.Tensor: Losses per training step
        """
        with torch.no_grad():
            random_T = t is None
            test_loss =  diffmodel.loss(test_data, random_T, t=t)
        if prev_tracks == None: 
            losses = test_loss.unsqueeze(0).detach()
        else: 
            losses = torch.cat((prev_tracks,  test_loss.unsqueeze(0).detach()), 0)
        return losses
    return track_test_loss


name_to_track_fn = {
    "loss": track_loss
}

class Tracker():
    """A class to track the evolution of variables during training.
    """
    def __init__(self, tracking_fns, tracking_fns_names): 
        """set up tracker

        Args:
            tracking_fns (list(str)): list of variables to be tracked. Currently available: 
            ["loss", "loss_theo", "eigvals_theo"]
        """
        self.tracking_fns = tracking_fns
        self.tracking_fns_names = tracking_fns_names
        self.tracks = {}
        for fn in self.tracking_fns_names: 
            self.tracks[fn] = None
        self.tracks["steps"] = []
            
    def track(self, diffmodel, step, x_0, **args):
        """Calls all tracking functions.

        Args:
            diffmodel (Diffusion1d): Diffusion Model
            step (int): training step number
        """
        for fn_name, fn in zip(self.tracking_fns_names, 
                               self.tracking_fns):
            self.tracks[fn_name] = fn(self.tracks[fn_name],
                                     diffmodel,
                                     step,
                                     x_0,
                                     **args)
        self.tracks["steps"] += [step]