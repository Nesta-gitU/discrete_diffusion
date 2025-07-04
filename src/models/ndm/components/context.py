import torch
from torch import nn
from wandb.proto.wandb_internal_pb2 import KeepaliveRequest

class VaeContext(nn.Module):
    def __init__(self,
                 model: nn.Module
                 ):
        super().__init__()
        
        self.model = model
        
    def forward(self, x): 
        #return the context and the loss from the context      
        #print(x) 
        mean, log_std = self.model(x).chunk(2, dim=-1)
        std = torch.exp(log_std)
        #print(mean.shape, "the shape of the mean")
        #print(log_std.shape, "the shape of the log_std")
        print(mean.mean(), "mean of the mean")
        print(std.mean(), "mean of the std")

        context = mean + std * torch.randn_like(std)
        #print(context.shape, "the shape of the context")

        KLD = -0.5 * (1 + (2*log_std) - mean**2 - (2*log_std).exp())
        
        return context, KLD
    
    def sample_context(self, x):
        bs = x.size(0)
        context = torch.randn(bs, self.model.output_dim, dtype=x.dtype, device=x.device)
        print("sampled context for the generations, this should happen only once")
        print(context.shape, "the shape of the context")
        return context

class EncoderContext(nn.Module):
    def __init__(self,
                 model: nn.Module
                 ):
        super().__init__()
        
        self.model = model
        
    def forward(self, x):       
        context = self.model(x)

        #loss is torch.zeros of bs
        loss = torch.zeros(x.size(0), x.size(1), dtype=context.dtype, device=x.device)
        
        return context, loss
    
    def sample_context(self, x):
        context = self.model(x)
        
        
        return context

class NoneContext(nn.Module):
    def __init__(self,
                 model
                 ):
        super().__init__()
        
    def forward(self, x):       
        return None, torch.zeros(x.size(0), dtype=x.dtype, device=x.device)

    def sample_context(self, x):
        return None