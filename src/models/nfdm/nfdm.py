import torch
import torch.autograd
import torch.autograd.functional
import torch.nn as nn

from their_utils.utils import token_discrete_loss


from pathlib import Path
import pickle

def jvp(f, x, v):
    #print(x.dtype)
    #print(v.dtype)
    with torch.inference_mode(False):
        x = x.clone()
        v = v.clone()
        #print(x.requires_grad, "x requires grad")
        #print(v.requires_grad, "v requires grad")
        return torch.autograd.functional.jvp(
            f, x, v, 
            create_graph=torch.is_grad_enabled()
        )

def t_dir(f, t):
    return jvp(f, t, torch.ones_like(t))

class NeuralDiffusion(nn.Module):
    def __init__(self, affine, pred, vol, diff_loss_type="elbo"):
        super().__init__()
        
        self.affine = affine
        self.pred = pred
        self.vol = vol
        self.diff_loss_type = diff_loss_type

        #self.encoder = encoder # instead of the encoder it should be like model.get_embedding() or something like that. #TODO
        #self.decoder = decoder #same goes for the decoder here. #TODO
        #acces the model by self.pred.model #affine.model is to be used only for the affine transformation. 
        #that means we do not need the lines below. 
        #if hasattr(self.decoder, "lm_head"):
        #    with torch.no_grad():
        #        self.decoder.lm_head.weight = self.encoder.embedding.weight

    def forward(self, x, t, **model_kwargs):
        return self.pred(x, t, **model_kwargs)
        
    def get_losses(self, x, t, 
                compute_diffusion_loss=True,
                compute_prior_loss=False,
                compute_reconstruction_loss=True,
                reconstruction_loss_type = "collapse",
                compute_their_loss=False,
                **model_kwargs
                ):

        def f(x_in):
            def f_(t_in):
                return self.affine(x_in, t_in)
            return f_

        bs = x.size(0)
        #t = torch.zeros_like(t) + 0.01

        #sanity check the input
        if compute_reconstruction_loss:
            if reconstruction_loss_type not in ["diff_anchor", "collapse"]:
                raise ValueError("Invalid reconstruction loss type")
        
        if not compute_diffusion_loss and not compute_reconstruction_loss:  
            raise ValueError("At least one of compute_diffusion_loss or compute_reconstruction_loss must be True")
        
        #encode the word indices x to embeddings
        #embeddings = self.encoder(x)
        embeddings = self.pred.model.get_embeds(x)
        # compute parameters of q(z_t | x) and corresponding time derivatives
        # that means here we get the parameters of: F(x, t, eps) = _F(e(x), t, eps)
        #here it crahses
        #print("t requires grad: ", t.requires_grad)
        #print("embeddings requires grad: ", embeddings.requires_grad)

        #require grad on the embeddings to make the jvp work correctly during test and val
        #print(t.requires_grad, "t requires grad")
        #print(embeddings.requires_grad, "embeddings requires grad")

        #self.affine(embeddings,t)
        if compute_diffusion_loss:
            g2 = self.vol(t) ** 2
            #print(self.affine(embeddings, t), "affine")
            (f_m, f_s, alpha), (f_dm, f_ds, alpha_prime) = t_dir(f(embeddings), t) #(function output), (jvp) == (mean, sigma), (mean derivative, sigma derivative)
            #print(f_dm, f_ds, "f_dm, f_ds")
        else:
            f_m, f_s, blank = self.affine(embeddings, t)

        # sample z_t from q(z_t | x)
        # z_t should obtained from putting epsilon into the forward process
        eps = torch.randn_like(embeddings)
        #fix epsilon to some value
        #eps = torch.ones_like(embeddings)/10
        z = f_m + f_s * eps # function evaluation of F(e(x), t, eps)

       

        
        if compute_diffusion_loss == "x_0":         #self, f_dm, f_ds, f_s, eps, g2, x_,           x, z, t
            embeddings_ = self.pred(z, t, **model_kwargs) # z is not neccerily a word embedding here.
            #print(embeddings, "embeddings")
            #print(embeddings_, "embeddings_")

            diffusion_loss = self.diffusion_loss(alpha, alpha_prime, f_s, f_dm, f_ds, eps, g2, embeddings_ , x, z, t, f)
        elif compute_diffusion_loss == "flow":
            # compute the drift term of the forward process based on eps
            f_dz = f_dm + f_ds * eps  # ODE drift
            f_score = - eps / f_s  # score function
            f_drift = f_dz - 0.5 * g2 * f_score  # SDE drift

            # predict f_drift from z_t
            r_drift = self.pred.model(z, t.squeeze(-1).squeeze(-1), **model_kwargs)
            
            # compute the diffusion loss
            diffusion_loss = 0.5 * (f_drift - r_drift) ** 2 / g2
            
        elif compute_their_loss:
            #put their mse loss here, then tune it until it initializes in exactly the same way with that loss, it should!
            #if do the loss below I replicate their loss precisely, but is this actually what diff loss for me does?
            #diffusion_loss = (embeddings - embeddings_) ** 2
            #now clearly this loss is not an ELBO (or its optimizable part), so lets add back the removed terms to see if it still works.
            embeddings_ = self.pred(z, t, **model_kwargs) # z is not neccerily a word embedding here.
            diffusion_loss = 0.5 * (embeddings - embeddings_) ** 2 / (f_s ** 2)

            #doing x_0 prediction with elbo and continuous time seems hard so maybe test it instead in their code. 
            
        else:
            diffusion_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)
        
        # compute the reconstruction loss
        if compute_reconstruction_loss:
            reconstruction_loss = self.reconstruction_loss(x, t, embeddings, token_discrete_loss)
        else:
            reconstruction_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device).mean()

        # compute the prior loss
        if compute_prior_loss:
            prior_loss = self.prior_loss(embeddings, t)
        else:
            prior_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)

        return diffusion_loss, reconstruction_loss, prior_loss

    def diffusion_loss(self, alpha, alpha_prime, f_s, f_dm, f_ds, eps, g2, x_, x, z, t, f):
        # compute the drift term of the forward process based on eps
        # f_drift is what is then used in the loss and is the forward process drift.
        f_dz = f_dm + f_ds * eps  # ODE drift ---> this works because gaussians are nice and linear so the derivative of F(x, t, eps) can be written like this. 
        f_score = - eps / f_s  # score function
        f_drift = f_dz - 0.5 * g2 * f_score  # SDE drift 

        #print(f_dm, "f_dm")

        #require grad because otherwise during val it wouldnt require grad and jvp wouldnt work right
        #x_.requires_grad = True
        
        # substitute predicted \hat{x} into the forward process to parameterise the reverse process
        
        (r_m, r_s, avalue), (r_dm, r_ds, anothervalue) = t_dir(f(x_), t)
        #print(r_dm, r_ds, "r_dm, r_ds")

        # compute the drift term of the reverse process based on z_t
        r_dz = r_dm + r_ds / r_s * (z - r_m)  # ODE drift
        r_score = (r_m - z) / r_s ** 2  # score function
        r_drift = r_dz - 0.5 * g2 * r_score  # SDE drift

        # not elbo
        #loss = (f_drift - r_drift) ** 2
        # ELBO
        #loss = 0.5 * (f_drift - r_drift) ** 2 / g2
        
        # SNR weighted Elbo
        #snr_prime = 2 * f_m * (f_dm * f_s - f_m * f_ds) / (f_s ** 3)
        #snr_prime = 2 * alpha * (alpha_prime * f_s - alpha * f_ds) / (f_s ** 3)
        #print(snr_prime, "snr_primenew")
        snr_prime = (-1/2) * ((0.9999*t + 0.000099) ** (-3/2)) * 0.9999

        if self.diff_loss_type == "elbo":
            loss = 0.5 * (f_drift - r_drift) ** 2 / g2
        
        elif self.diff_loss_type == "x_0_prediction":
            loss = (1/(-0.5 * snr_prime)) * 0.5 * (f_drift - r_drift) ** 2 / g2
        else:
            raise ValueError("Invalid diffusion loss type")

    
        # mask out special tokens
        #mask = x != 0 #false for [UNK] tokens which should have id zero. 
        #not_mask = ~mask
        #print(not_mask.sum(), "number of unks")
        #mask_expanded = mask.unsqueeze(-1).expand(-1, -1, loss.shape[2])
        #loss = loss * mask_expanded
        
        #do not comment out this line!!!!!!!!!!!!!!!!
        #loss = loss.sum(dim=(1,2))

        return loss




    def reconstruction_loss(self, x, t, embeddings, token_discrete_loss):
        """
        I need to have a good look at the math of this part. 
        """
        
        f_m, f_s, blank = self.affine(embeddings, torch.zeros_like(t))
        z_0 = f_m + f_s * torch.randn_like(embeddings)

        decoder_nll = token_discrete_loss(z_0, self.pred.model.get_logits, x)
        
        return decoder_nll

    #this should only be computed if the forward process isnt implemented to have exactly n(0,1) at time 1. 
    def prior_loss(self, x, t):
        # compute the prior loss 
        #not implemented error
        mean, _, _ = self.affine(x, torch.ones_like(t))
        #print(mean, "mean")
        #print(variance, "variance")
        
        return mean ** 2
        
        
    

