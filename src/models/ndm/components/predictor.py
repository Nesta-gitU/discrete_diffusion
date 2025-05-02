from tabnanny import check
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
        if context is not None:
            #concatenate context with z
            #check if context is the same shape as one vector in z
            if context.shape[1] != z.shape[-1]:
                raise ValueError("context and z must have the same hidden_dim")
            z = torch.cat([z, context], dim=1)       
        x = self.model(z, t.squeeze(-1).squeeze(-1), **model_kwargs) 

        if self.stabilize:
            x = (1 - t) * z + (t + 0.01) * x
        
        return x