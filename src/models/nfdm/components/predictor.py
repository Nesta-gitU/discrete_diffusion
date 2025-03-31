import torch
from torch import nn

#this class corresponds to x_theta(z_t, t), I will decouple it so that the predicted x can also be use elsewhere. in my case it will be predicted e but okay. 
class Predictor(nn.Module):
    def __init__(self,
                 model: nn.Module):
        super().__init__()
        
        self.model = model
        print("the predictor model is", model)
        
    def forward(self, z, t, **model_kwargs):
        #z_t = torch.cat([z, t], dim=1)
        #x = self.model(z_t)        
        x = self.model(z, t.squeeze(-1).squeeze(-1), **model_kwargs) # t conditioning not implemented yet in the model
        
        #with torch.no_grad():
            #out1 = self.model(x, t)
            #out2 = self.model(x, t)
            #print("out1", out1)
            #print("out2", out2)
            #print("next iteration")
        #print(torch.allclose(out1, out2, atol=1e-6), "what is going on this should be true")

        # Reparametrization for numerical stability
        #print(t.shape, "t")
        #print(z.shape, "z")
        #print(x.shape, "x")
        #x = (1 - t) * z + (t + 0.01) * x
        #they dont do this in their paper so lets not do it here#TODO
        
        return x