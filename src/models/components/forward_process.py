import torch
from torch import nn
import numpy as np

#not yet created
from nets.net import Net

class NFDM_gaussian(nn.Module):
    """
    Affine transformation module. It parametrizes the forward
    process of the diffusion model with a Gaussian distribution.

    F(x, t, eps) = \mu(x, t) + \sigma(x, t) * eps
    """

    def __init__(self,
                 model: nn.Module):
        super().__init__()
        
        #the net used to predict the mean and std_bar in the gaussian parameterization, with as input x and t, and as ouput mu and log(sigma)
        self.net = model
        
    def forward(self, x, t):
        # Conditional optimal transport (FM-OT)
        # return (1 - t) * x, t + (1 - t) * 0.01

        # Learnable Gaussian forward process (NFDM-Gaussian)
        x_t = torch.cat([x, t], dim=1)
        m_ls = self.net(x_t) 
        m, ls = m_ls.chunk(2, dim=1)

        m = (1 - t) * x + t * (1 - t) * m #m is mu_hat
        ls = (1 - t) * np.log(0.01) + t * (1 - t) * ls #ls is log(sigma_hat) so the final expresion is log sigma. 
        #0.01 is delta, so delta^2 = 0.0001 like in the paper

        return m, torch.exp(ls)

class FM_OT(nn.Module):
    """
    og flow matching parameterization
    """

    def __init__(self):
        super().__init__()
        
    def forward(self, x, t):
        # Conditional optimal transport (FM-OT)
        return (1 - t) * x, t + (1 - t) * 0.01

        