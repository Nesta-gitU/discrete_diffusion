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
        
    def forward(self, z, t, context=None, **model_kwargs):       
        x = self.model(z, t.squeeze(-1).squeeze(-1), **model_kwargs) 

        value = True if hasattr(self, "stabilize") else False

        if value:
            if self.stabilize:
                x = (1 - t) * z + (t + 0.01) * x
                #x = 1/(1-t) * z - (t + 0.01) * x # this formulation fails to take into account the end points 0 and 1 which makes it unstable 
                #x = (1 + t) * z - (t + 0.01) * x
        
        return x