import torch
from torch import nn
import numpy as np

from .nets.mlp import MLP

class NFDM_gaussian(nn.Module):
    """
    Affine transformation module. It parametrizes the forward
    process of the diffusion model with a Gaussian distribution.

    F(x, t, eps) = \mu(x, t) + \sigma(x, t) * eps
    """

    def __init__(self,
                 model: nn.Module,
                 dont_add_head=False):
        super().__init__()
        
        #the net used to predict the mean and std_bar in the gaussian parameterization, with as input x and t, and as ouput mu and log(sigma)
        self.net = model
        self.output_dim = 128
        self.dont_add_head = dont_add_head
        if dont_add_head:
            pass
        else:
            self.linear_layer = nn.Sequential(nn.Linear(self.output_dim, int(self.output_dim * 4)), 
                                            nn.ReLU(), 
                                            nn.Linear(int(self.output_dim * 4), int(self.output_dim*4)),
                                            nn.ReLU(),
                                            nn.Linear(int(self.output_dim*4), self.output_dim))	

            self.linear_layer2 = nn.Sequential(nn.Linear(self.output_dim, int(self.output_dim * 4)), 
                                            nn.ReLU(), 
                                            nn.Linear(int(self.output_dim * 4), int(self.output_dim*4)),
                                            nn.ReLU(),
                                            nn.Linear(int(self.output_dim*4), self.output_dim))	
        
    def forward(self, x, t):
        #print("using NFDM-Gaussian")
        # Conditional optimal transport (FM-OT)
        #return (1 - t) * x, t + (1 - t) * 0.1, torch.tensor(0.0).to(x.device)

        # Learnable Gaussian forward process (NFDM-Gaussian)
        #x_t = torch.cat([x, t], dim=1)
        #m_ls = self.net(x_t)  
        #TODO: t conditioning not implemented yet 0.01
        #small_value = torch.log(torch.sqrt(torch.sigmoid(torch.tensor(-10))))
        #small_value2 = torch.log(torch.sqrt(torch.sigmoid(torch.tensor(10))))
        small_value = np.log(0.1) #i I make it 0.01 it gets very low loss but that implies t=0 -> s = exp(0.01) = 1.01	which is clearly wrong 
        #if torch.all(t == torch.ones_like(t)):
        #    m = torch.zeros_like(x)
        #    ls = torch.zeros_like(x)
        #print("using this code------------------------------------------------------------")
        if torch.all(t == torch.zeros_like(t)):
            m = x
            ls = (1 - t) * small_value      # * torch.ones_like(x)
        else:
            m_ls = self.net(x, t.squeeze(-1).squeeze(-1)) 
            m, ls = m_ls.chunk(2, dim=2)    #why was this 1 before 
            #ls = ls.clamp(min=-20, max=10)  #clamping

            #make the variance tokenwise instead of dimensionwise
            value = True if hasattr(self, 'dont_add_head') else False
            if value:
                if not self.dont_add_head:
                    ls = self.linear_layer(ls)
                    m = self.linear_layer2(m)
            else:
                ls = self.linear_layer(ls)
                m = self.linear_layer2(m)
            #ls = ls.mean(dim=2, keepdim=True)
            #print("hello", ls.shape)

            m = (1 - t) * x + t * (1 - t) * m   #m is mu_hat -> (1 - t) * x = z - t * (1 - t) * x  -> x = 1/(1-t) * z - t * x
            ls = (1 - t) * small_value + t * (1 - t) * ls  #ls is log(sigma_hat) so the final expresion is log sigma. 
        #0.01 is delta, so delta^2 = 0.0001 like in the paper
        
        return m, torch.exp(ls), torch.tensor(0.0).to(x.device)

class ndm(nn.Module):
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
        s = (0.99-t) * 0.0001
        alpha = torch.sqrt(1- torch.sqrt(t+s))
        sigma = torch.sqrt(torch.sqrt(t + s))

        transformation = self.net(x, t.squeeze(-1).squeeze(-1))

        mean = alpha * transformation
        return mean, sigma, alpha 


class FM_OT(nn.Module):
    """
    og flow matching parameterization
    """

    def __init__(self):
        super().__init__()
    
    def forward(self, x, t):
        #print("using FM-OT")
        # Conditional optimal transport (FM-OT)

        # a seperate output for the diffusion loss scaling term
        output3 = (1-t)

        output0 = output3 * x
        output1 = t + (1 - t) * 0.01
        #print("output0 requires grad: ", output0.requires_grad)
        #print("output1 requires grad: ", output1.requires_grad)


        return output0, output1, output3

class Sqrt(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, t):
        # Scale t to simulate a different schedule
        s = (0.99-t) * 0.0001
        output3 = torch.sqrt(1- torch.sqrt(t+s))

        output0 = output3 * x
        output1 = torch.sqrt(torch.sqrt(t + s))
        #print("1is it using this code=------------------------------------------------------------")

        #a seperate output for the diffusion loss scaling term
        
        return output0, output1, output3

"""
class Sqrt2(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, t):
        # Sce t to simulate a different schedule
        print("2or is it using this code------------------------------------------------------------")
        s = (0.99-t) * 0.0001
        output0 = (torch.sqrt(1 - torch.sqrt(t + s)) - torch.sqrt(1 - torch.sqrt(torch.tensor(1.0).to(t.device) + (0.99-1) * 0.0001)))* x
        output1 = torch.sqrt(torch.sqrt(t + s))
        return output0, output1
"""

'''
class Sqrt(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, t):
        # Scale t to simulate a different schedule
        #s = (0.99-t) * 0.0001
        t = t * 2000
        output0 = torch.sqrt(1 - torch.sqrt(t/2000)) * x
        output1 = torch.sqrt(torch.sqrt(t/2000 + 0.0001))
        return output0, output1

'''

class SigmoidApprox(nn.Module):
    def __init__(self, shift=0.28, scale=0.12):
        """
        shift: moves the sigmoid left/right.
        scale: controls the steepness of the sigmoid.
        A large negative shift means that for most t in [0,1] the sigmoid is nearly zero,
        and then it quickly transitions to 1.
        """
        super().__init__()
        self.shift = shift
        self.scale = scale

    def forward(self, x, t):
        print("using sigmoidapprox")
        # Assume t is scaled between 0 and 1
        s = torch.sigmoid((t - self.shift) / self.scale) + 0.0025
        # For t values where s is near zero, output0 ~ x (i.e. very little noise added)
        # and output1 ~ 0. As t increases, s moves toward 1, so output0 shrinks and
        # output1 increases, similar to a noise schedule.
        output3 = ((1 - s) + ((1-t) * 0.06))
        output0 = output3 * x
        output1 = s
        return output0, output1, output3
        

        