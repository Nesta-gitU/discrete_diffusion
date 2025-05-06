from torch import nn
import torch
from .nets.mlp import MLP

class LearnedVolatility(nn.Module):
    """
    Volatility module. It parametrizes the volatility g of the reverse and forward processes.

    d z = f(z, t) d t + g(t) d \bar{w}_t
                        ^^^^
    """
    def __init__(self):
        super().__init__()
        
        self.model = MLP(n_inputs= 1, n_hidden= [64, 64, 64], n_outputs=1,use_batch_norm=False)
        #make sure the model has ouput and input dim of 1!!!!!

        self.sp = nn.Softplus()
        
    def forward(self, t):
        # Volatility, that corresponds to linear log-SNR schedule in DDPM/VDM
        # return (20 * torch.sigmoid(-10 + 20 * t)) ** 0.5

        # Learnable volatility
        return self.sp(self.model(t))


class StaticVolatility(nn.Module):
    """
    Volatility module. It parametrizes the volatility g of the reverse and forward processes.

    d z = f(z, t) d t + g(t) d \bar{w}_t
                        ^^^^
    """
    def __init__(self):
        super().__init__()
 
    def forward(self, t):

        # Volatility, that corresponds to linear log-SNR schedule in DDPM/VDM
        return (20 * torch.sigmoid(-10 + 20 * t)) ** 0.5

class SqrtVolatility(nn.Module):
    """
    Volatility module. It parametrizes the volatility g of the reverse and forward processes.

    d z = f(z, t) d t + g(t) d \bar{w}_t
                        ^^^^
    """
    def __init__(self):
        super().__init__()
 
    def forward(self, t):
        sigma2 = torch.sqrt(t + (0.99 - t) * 0.0001)
        g2 = 0.9999 / (2 * sigma2 * (1-sigma2))
        return torch.sqrt(g2)
