import torch
import numpy as np
import torch.nn.functional as F
# import lpips
import torch.nn as nn
from einops import rearrange
from functools import partial
import math


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))

def sum_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.sum(x, dim=list(range(1, len(x.size()))))
    

class SiTLoss:
    def __init__(
            self,
            prediction='v',
            path_type="linear",
            weighting="uniform",
            stochastic=False,
            noise_scale=0.01,
            ):
        self.prediction = prediction
        self.weighting = weighting
        self.path_type = path_type
        self.stochastic = stochastic
        self.noise_scale = noise_scale
        
    def consistency(self, zt, v, t):
        if self.path_type == "linear":
            # v = z1 - z0
            z1 = zt + (1 - t) * v
            z0 = zt + (0 - t) * v
        elif self.path_type == "cosine":
            # v = dz/dt = d_alpha*z0 + d_sigma*z1
            theta = t * (math.pi / 2)
            alpha = torch.cos(theta)
            sigma = torch.sin(theta)
            z1 = sigma * zt + (2 / math.pi) * alpha * v
            z0 = alpha * zt - (2 / math.pi) * sigma * v
        return z1, z0

    def interpolant(self, t):
        if self.path_type == "linear":
            alpha_t = 1 - t
            sigma_t = t
            d_alpha_t = -1
            d_sigma_t =  1
        elif self.path_type == "cosine":  
            alpha_t = torch.cos(t * np.pi / 2)
            sigma_t = torch.sin(t * np.pi / 2)
            d_alpha_t = -np.pi / 2 * torch.sin(t * np.pi / 2)
            d_sigma_t =  np.pi / 2 * torch.cos(t * np.pi / 2)
        else:
            raise NotImplementedError()

        return alpha_t, sigma_t, d_alpha_t, d_sigma_t

    def __call__(self, model, images, fmri):
        
        # sample timesteps
        if self.weighting == "uniform":
            time_input = torch.rand((images.shape[0], 1, 1))
        elif self.weighting == "lognormal":
            # sample timestep according to log-normal distribution of sigmas following EDM
            rnd_normal = torch.randn((images.shape[0], 1 ,1))
            sigma = rnd_normal.exp()
            if self.path_type == "linear":
                time_input = sigma / (1 + sigma)
            elif self.path_type == "cosine":
                time_input = 2 / np.pi * torch.atan(sigma)
        
        #! reverse z0 and z1
        z0 = images
        z1 = fmri
        
        time_input = time_input.to(images.device)
        alpha_t, sigma_t, d_alpha_t, d_sigma_t = self.interpolant(time_input)
        
        zt = alpha_t * z0 + sigma_t * z1  # zt = (1-t) * z0 + t * z1  
        
        if self.stochastic:
            noise = torch.randn_like(zt) * (self.noise_scale * torch.sin(np.pi * time_input))
            zt = zt + noise
        
        target = d_alpha_t * z0 + d_sigma_t * z1    # target = -z0 + z1 = z1 - z0
        
        score = model(zt, time_input.flatten())
        denoising_loss = mean_flat((score - target) ** 2).mean()

        return denoising_loss