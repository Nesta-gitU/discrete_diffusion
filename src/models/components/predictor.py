import torch
from torch import nn

#this class corresponds to x_theta(z_t, t), I will decouple it so that the predicted x can also be use elsewhere. in my case it will be predicted e but okay. 
class Predictor(nn.Module):
    def __init__(self,
                 model: nn.Module):
        super().__init__()
        
        self.model = model
        
    def forward(self, z, t):
        #z_t = torch.cat([z, t], dim=1)
        #x = self.model(z_t)
        
        x = self.model(z, t) # t conditioning not implemented yet in the model

        # Reparametrization for numerical stability
        x = (1 - t) * z + (t + 0.01) * x
        
        return x