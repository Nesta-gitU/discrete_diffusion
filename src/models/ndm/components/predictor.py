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
            #print("Using context in predictor")
            #print(context, "context")
            #concatenate context with z
            #check if context is the same shape as one vector in z
            if context.shape[1] != z.shape[-1]:
                raise ValueError("context and z must have the same hidden_dim")
            #unsqueeze the middel dimension of context
            context = context.unsqueeze(1)
            z = torch.cat([context, z], dim=1)       
        x = self.model(z, t.squeeze(-1).squeeze(-1), **model_kwargs) 

        #dont actually need the first token
        #print(x.shape)
        #print(z.shape)
        if context is not None:
            x = x[:, 1:, :]
            z = z[:, 1:, :]
        
        if self.stabilize:
            x = (1 - t) * z + (t + 0.01) * x
        
        
        return x