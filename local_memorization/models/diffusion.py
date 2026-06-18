import torch
from tqdm import tqdm
from .betas import linear, linear_scaled
from local_memorization.models.tracker import Tracker, track_loss
from local_memorization.models.networks import AffineLinear
import numpy as np


class Diffusion: 
    def __init__(self, model, dim, device, T=1000, c_reg=0, beta=None): 
        """general purpose diffusion model based on Refs [1,2]

        Args:
            model (torch.nn): the noise predictor
            dim (int): data dimension (flat)
            device (torch.device): device (cuda or cpu)
            T (int, optional): length of noising process. Defaults to 1000.
        """
        # basic attributes
        self.model = model #network to be trained
        self.dim = dim
        self.device = device
        self.T = T

        # noising schedule
        self.ts = torch.arange(T)
        if beta is None:
            if T == 1000:
                self.beta = linear(T=T).to(device)
            else:
                self.beta = linear_scaled(T=T).to(device)
        else: 
            assert(len(beta)) == T
            self.beta = beta.to(device)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.sigma = torch.sqrt(self.beta)
        self.gamma = self.alpha_bar*c_reg

        # for training
        self.loss_func = torch.nn.MSELoss()
        self.optimizer = None
        self.tracker = None # stores variables such as loss during training
        self.l_t = torch.ones_like(self.beta)
        self.steps = 0
        # post processing of a sample
        self.post_process = lambda sample: sample

    def multiply_with_time_factor(self, time_factor, x): 
        """multiply tensor with factor varying along
          first dimension.

        Args:
            time_factor (torch.Tensor): muliplication factor
            x (torch.tensor): factor to multiply

        Raises:
            ValueError: Works only for tensors of 2 or 4 dimensions

        Returns:
            torch.tensor: outcome of multiplication
        """
        if time_factor.dim() == 0: 
            return time_factor*x 
        else:
            assert len(time_factor) == x.shape[0]
            if len(x.shape) == 2: 
                time_factor = time_factor.reshape(-1,1)
            elif len(x.shape) == 4: 
                time_factor = time_factor.reshape(-1,1,1,1)
            else: 
                raise ValueError(f"x has shape {x.shape}, but expected one of (num_samples, dim)"
                                +"or (num_samples, num_channels, dim1d, dim1d)")
            return time_factor*x

    def add_forward_noise(self, x_0, t):
        """add noise according to schedule

        Args:
            x_0 (torch.tensor): data
            t (torch.tensor): time in forward process

        Returns:
            torch.tensor, torch.tensor: noised data, noise
        """
        noise = torch.randn(x_0.shape).to(self.device)
        pref_x_0 = self.sqrt_alpha_bar[t]
        pref_noise = self.sqrt_one_minus_alpha_bar[t]
        noised_data = self.multiply_with_time_factor(pref_x_0, x_0) \
                + self.multiply_with_time_factor(pref_noise, noise) 
        return noised_data, noise
    
    def sample_noise(self, u_t, t): 
        """generate noise for sampling

        Args:
            u_t (torch.tensor): current sample, needed only for shape
            t (torch.tensor): time in backward process

        Returns:
            torch.tensor: noise, scaled by backward sqrt(variance) sigma
        """
        noise = torch.randn(u_t.shape).to(self.device)
        return self.multiply_with_time_factor(self.sigma[t], noise)
    
    def mu_theta(self, x_t, t, style="Ho"): 
        """mean of the backward process according to 
        Ref[1].

        Args:
            x_t (torch.tensor): data
            t (torch.tensor): time in forward noising process

        Returns:
            torch.tensor: predicted mean
        """
        epsilon_theta = self.model(x_t, t)
        pref_epsilon = - (self.beta[t]
                          / self.sqrt_one_minus_alpha_bar[t])
        if style == "Ho":
            pref_epsilon = - (self.beta[t]
                          / self.sqrt_one_minus_alpha_bar[t])
        elif style == "Song": 
            pref_epsilon = - self.sqrt_one_minus_alpha_bar[t]
        term_2 = self.multiply_with_time_factor(pref_epsilon,
                                                epsilon_theta)
        
        return self.multiply_with_time_factor(1.0/torch.sqrt(self.alpha[t]),
                                               x_t + term_2)


    
    def sample(self, num_samples, t_min=0,
               latents=None, shape=None, style="Ho", sample_batch=False):
        """generate samples

        Args:
            num_samples (int): number of samples
            t_min (int, optional): time in backward process
                    at which to sample. Defaults to 0.
            latents (torch.tensor, optional): initial noise tensor. Defaults to None.
            shape (tuple, optional): Shape of samples, must match model input. 
                Defaults to None, which corresponds to flat data of shape (#samples, dim    )
            style (str, optional): Whether to use sampling from Ref [1]("Ho") 
                or Ref[2] ("Song"). Defaults to "Ho".

        Returns:
            _type_: _description_
        """
        if shape is None: 
            shape = (num_samples, self.dim)

        if latents is None: 
            latents = torch.randn(shape).to(self.device)
            u_t = latents
        else: 
            u_t = latents.clone().detach()

        if not sample_batch:
            sample_batch = num_samples
            number_of_batches = 1
        else: 
            number_of_batches = int(np.ceil(num_samples*1.0/sample_batch))

        ts = list(range(t_min,self.T))
        ts.reverse()
        with torch.no_grad():
            for i in range(number_of_batches):
                u_t_batch = latents[i*sample_batch:(i+1)*sample_batch, :].clone().detach()
                for t in tqdm(ts,"sampling loop"): 
                    t_vec = (torch.ones(u_t_batch.shape[0])*(t)).long().to(self.device)
                    rhs = self.mu_theta(u_t_batch,
                                        t_vec,
                                        style=style) \
                        + self.sample_noise(u_t_batch,
                                             t_vec)
                    if style == "Song": 
                        epsilon_theta = self.model(u_t_batch,
                                                    t_vec)
                        rhs += self.multiply_with_time_factor( \
                                    torch.sqrt(1 - self.alpha_bar[t_vec]/self.alpha[t_vec]
                                            - self.sigma[t_vec]**2),
                                    epsilon_theta)
                    u_t_batch = rhs
                u_t[i*sample_batch:(i+1)*sample_batch] = u_t_batch
        return self.post_process(u_t)
    
    def draw_ts(self, x_0, stack_samples, t=None): 
        """Draw denoising time steps at random.

        Args:
            x_0 (torch.tensor): data
            stack_samples (bool, optional): Whether to use more than one noise 
                realization per sample in batch. If int, will use that number 
                of noise realizations in total. E.g. if batch_size is 10 but 
                stack-samples is 100, will use 10 noise realizations per sample.
                Defaults to False.
            t (int, optional): If not None, will return only that
                denoising time point. Defaults to None.

        Returns:
            x_0: ...
            ts: random times
        """
        if stack_samples:
            num_xs = x_0.shape[0]
            if num_xs < stack_samples:
                x_0 = torch.vstack([x_0,]*int(stack_samples/num_xs))
        t_len = x_0.shape[0]
        if t is None:
            ts = torch.randint(0, self.T, (t_len,)).to(self.device)
        else: 
            ts = (torch.ones(t_len)*t).long()
        return x_0, ts
    
    def loss(self, x_0, stack_samples=False, t=None, ts=None, on_model=None): 
        """Compute loss (referred to as L_1 in Ref [2]),
        used in both refs.

        Args:
            x_0 (torch.tensor): data
            stack_samples (bool, optional): Whether to use more than one noise 
                realization per sample in batch. If int, will use that number 
                of noise realizations in total. E.g. if batch_size is 10 but 
                stack-samples is 100, will use 10 noise realizations per sample.
                Defaults to False.
            t (int, optional): If not None, will evaluate loss on only that
                denoising time point. Defaults to None.
        Returns:
            float: loss
        """
        if ts is None:
            x_0, ts = self.draw_ts(x_0, stack_samples, t=t)
        x_noised, noise = self.add_forward_noise(x_0, ts)

        if on_model is not None and type(t)==int: 
            epsilon_theta = on_model(x_noised)
        elif on_model is not None and t is None: 
            epsilon_theta = on_model(x_noised, ts)
        else:
            epsilon_theta = self.model(x_noised, ts)

        _loss = self.loss_func(epsilon_theta, noise)
        
        if type(self.model) == AffineLinear: 
            reg = torch.einsum("t, tij ->",
                               self.gamma[ts]/(self.dim*len(ts)), # divide to have same prefactor as MSEloss
                               self.model.weights(ts)**2)
            _loss += reg
        return _loss
    
    def train(self, dataloader, n_epochs=10, lr=1e-3,
              momentum=0.0, use_scheduler=False,
              track_steps=100, stack_samples=False,
              Adam=True, maxstep_per_epoch=None, t=None): 
        """train diffusion model.

        Args:
            dataloader (torch.utils.DataLoader): data loader
            n_epochs (int, optional): Number of epochs. Defaults to 10.
            lr (float, optional): learning rate. Defaults to 1e-3.
            momentum (float, optional): Momentum for Adam. Defaults to 0.0.
            use_scheduler (bool, optional): Whether to use a scheduler.
                Defaults to False.
            track_steps (int, optional): Optimization step interval after which 
                to track results. Defaults to 100.
            stack_samples (bool, optional): Whether to use more than one noise 
                realization per sample in batch. If int, will use that number 
                of noise realizations in total. E.g. if batch_size is 10 but 
                stack-samples is 100, will use 10 noise realizations per sample.
                Defaults to False.
            Adam (bool, optional): Whether to use Adam or SGD as optimizer. 
                Defaults to True.
            maxstep_per_epoch (int, optional): Maximum number of steps per epoch
              (if dataset is very large, allows for training for less than one 
              epoch). Defaults to None.
            t (int, optional): If not None, will train only that denoising time 
                point. Defaults to None.
        """
        # set up optimizer  
        self.lr = lr
        if self.optimizer is None:
            if Adam:
                self.optimizer = torch.optim.Adam(self.model.parameters(),
                                            lr=lr)
            else:
                self.optimizer = torch.optim.SGD(self.model.parameters(),
                                            lr=lr, momentum=momentum)

        # set up tracker (to track results during training, by default tracks
        # only training loss)
        if self.tracker is None:
            track_fns = [track_loss]
            track_fns_names = ["loss"]
            self.tracker = Tracker(track_fns, track_fns_names)

        # training loop        
        for _ in range(n_epochs): 
            for batch in tqdm(dataloader,"training loop"): # iterate over batches of data
                batch = batch[0].to(self.device) # zeroth element as dataloader returns list
                _loss = self.loss(batch, stack_samples=stack_samples, t=t)
                self.optimizer.zero_grad()
                _loss.backward()
                self.optimizer.step()
                if self.steps % track_steps==0: 
                    self.tracker.track(self, self.steps, batch, loss=_loss) 
                self.steps += 1
                if not maxstep_per_epoch is None: 
                    if self.steps >= maxstep_per_epoch:
                        break
        
        self.tracker.track(self, self.steps, batch, loss=_loss)

    def remove_regularization(self,): 
        self.gamma = torch.zeros_like(self.beta).to(self.device)


class DiffusionOnData(Diffusion): 
    def loss(self, x_0, stack_samples=False, t=None, ts=None, on_model=None): 
        """Compute loss, predicting the data instead of the noise.

        Args:
            x_0 (torch.tensor): data
            stack_samples (bool, optional): Whether to use more than one noise 
                realization per sample in batch. If int, will use that number 
                of noise realizations in total. E.g. if batch_size is 10 but 
                stack-samples is 100, will use 10 noise realizations per sample.
                Defaults to False.
            t (int, optional): If not None, will evaluate loss on only that
                denoising time point. Defaults to None.
        Returns:
            float: loss
        """
        if ts is None:
            x_0, ts = self.draw_ts(x_0, stack_samples, t=t)
        x_noised, noise = self.add_forward_noise(x_0, ts)

        if on_model is not None and type(t)==int: 
            mu_theta = on_model(x_noised)
        elif on_model is not None and t is None: 
            mu_theta = on_model(x_noised, ts)
        else:
            mu_theta = self.model(x_noised, ts)

        _loss = self.loss_func(mu_theta, x_0)
        
        if type(self.model) == AffineLinear: 
            reg = torch.einsum("t, tij ->",
                               self.gamma[ts]/(self.dim*len(ts)), # divide to have same prefactor as MSEloss
                               self.model.weights(ts)**2)
            _loss += reg
        return _loss
    
    def sample(self, num_samples, t_min=0, latents=None, shape=None, style="Ho", sample_batch=False):
        """Not implemented because it's different for models predicting the data
        (possible, just raises an error to prevent accidental use of the false
        learning routine)

        Args:
            num_samples (_type_): _description_
            t_min (int, optional): _description_. Defaults to 0.
            latents (_type_, optional): _description_. Defaults to None.
            shape (_type_, optional): _description_. Defaults to None.
            style (str, optional): _description_. Defaults to "Ho".
            sample_batch (bool, optional): _description_. Defaults to False.

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError