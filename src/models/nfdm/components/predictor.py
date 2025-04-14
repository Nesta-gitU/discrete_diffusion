import torch
from torch import nn

class Predictor(nn.Module):
    def __init__(self,
                 model: nn.Module,
                 stabilize: bool = False,
                 ):
        super().__init__()
        
        self.model = model
        self.stabilize = stabilize  
        
    def forward(self, z, t, **model_kwargs):       
        x = self.model(z, t.squeeze(-1).squeeze(-1), **model_kwargs) 

        if self.stabilize:
            x = (1 - t) * z + (t + 0.01) * x
        
        return x