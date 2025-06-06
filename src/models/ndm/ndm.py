from abc import ABC, abstractmethod
from multiprocessing import context
from typing import Callable, Optional

import numpy as np

import torch
from torch import embedding, nn, Tensor, zeros_like
from torch.nn import functional as F
import torch.distributions as D

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from src.models.ndm.components.gamma import Gamma
from src.models.ndm.components.transform import AffineTransform
from src.models.ndm.components.vol_eta import VolatilityEta
from src.models.ndm.components.predictor import Predictor
from src.models.ndm.components.context import NoneContext

from their_utils.utils import token_discrete_loss
 


class NeuralDiffusion(nn.Module):
    def __init__(self, transform: AffineTransform, gamma: Gamma, vol_eta: VolatilityEta, pred: Predictor, context: NoneContext = NoneContext(None), clamp_max=10000, diff_loss_type = "elbo", gamma_init=False, 
                 add_pure_x_pred=False):
        super().__init__()

        self.transform = transform
        self.gamma = gamma
        self.add_pure_x_pred = add_pure_x_pred
        self.context = context

        if clamp_max == "inf":
            clamp_max = float("inf")
        self.clamp_max = clamp_max
        if gamma_init:
            gamma.load_state_dict(
                        torch.load("src/models/ndm/gamma_checkpoints/vdm_checkpoint.pth", map_location="cpu"))

        self.vol_eta = vol_eta
        self.pred = pred

        self.diff_loss_type = diff_loss_type

        self.alpha_0 = None
        self.sigma_0 = None
        self.alpha_1 = None

        if diff_loss_type == "elbo_noise_scaling":
            self.scalar = nn.Parameter(torch.tensor([1.0]))
        
        print("----------------the used diff loss type is: ", diff_loss_type, "----------------")

    def forward(self, x: Tensor, t: Tensor, **model_kwargs):
        return self.pred(x, t, **model_kwargs)

    def get_losses(self, x: Tensor, t: Tensor,
                compute_diffusion_loss=True,
                compute_prior_loss=False,
                compute_reconstruction_loss=True,
                reconstruction_loss_type = "collapse",
                compute_their_loss=False, #this wont be used here. 
                **model_kwargs):
        
        bs = x.size(0)
        pad_mask = x == 3
        embeddings = self.pred.model.get_embeds(x)    
        if compute_their_loss:
            #put their mse loss here, then tune it until it initializes in exactly the same way with that loss, it should!
            #if do the loss below I replicate their loss precisely, but is this actually what diff loss for me does?
            #
            #now clearly this loss is not an ELBO (or its optimizable part), so lets add back the removed terms to see if it still works.
            context, context_loss = self.context(embeddings)

            eps = torch.randn_like(embeddings)

            if context is None:
                gamma, d_gamma = self.gamma(t)
            else:
                gamma, d_gamma = self.gamma(t, context)

            alpha = self.gamma.alpha_2(gamma) ** 0.5
            sigma = self.gamma.sigma_2(gamma) ** 0.5

            (m, _), (d_m, _) = self.transform(embeddings, t)

            eta = self.vol_eta(t)

            z = alpha * m + sigma * eps

            embeddings_ = self.pred(z.detach(), t, context, **model_kwargs) # z is not neccerily a word embedding here.
            diffusion_loss = (embeddings - embeddings_) ** 2

            diffusion_loss_full_elbo = None

        else:
            diffusion_loss, context_loss, diffusion_loss_full_elbo, embeddings_, _, _, _ = self.get_diffusion_loss(embeddings, t, pad_mask)

        if compute_reconstruction_loss:
            if reconstruction_loss_type == "collapse":
                reconstruction_loss = self.reconstruction_loss(x, t, embeddings)
            elif reconstruction_loss_type == "diff_anchor":
                reconstruction_loss = self.reconstruction_loss(x, t, embeddings_)
        else:
            reconstruction_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device).mean()

        # compute the prior loss
        if compute_prior_loss:
            prior_loss = self.prior_loss(embeddings, t)
        else:
            prior_loss = torch.zeros(bs, dtype=embeddings.dtype, device=x.device)
        
        return diffusion_loss, context_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss

            
        
    def get_diffusion_loss(self, x: Tensor, t: Tensor, pad_mask: Tensor):
        
        context, context_loss = self.context(x)

        eps = torch.randn_like(x)

        if context is None:
            gamma, d_gamma = self.gamma(t)
        else:
            gamma, d_gamma = self.gamma(t, context)

        alpha = self.gamma.alpha_2(gamma) ** 0.5
        sigma = self.gamma.sigma_2(gamma) ** 0.5

        (m, _), (d_m, _) = self.transform(x, t)

        eta = self.vol_eta(t)

        z = alpha * m + sigma * eps

        x_ = self.pred(z, t, context)

        (m_, _), (d_m_, _) = self.transform(x_, t)

        #so I would like to weight like x prediction so I should use this weighting instead
        #the l_x weighting above still includes the 1/2g(t) term but in my other formulation it actually doesnt, since g(t) >1? is that true, that could make the loss a bit lower
        #okay I take that back
        one_over_dgamma = torch.clamp(1 / (d_gamma), max=self.clamp_max) #so it doesnt go to inf 

        loss_1 = (1 + eta) / 2 * (m - m_) + one_over_dgamma * (d_m - d_m_)
        loss_1 = loss_1 ** 2
        loss = loss_1

        # Stabilises training
        if self.add_pure_x_pred:
            loss_x = (1 + eta) ** 2 / 4 * (x - x_) ** 2
            loss = 0.5 * loss + 0.5 * loss_x

        if self.diff_loss_type == "elbo":
            # ELBO weighting
            lmbd_elb = 0.5 * torch.exp(-gamma) * d_gamma / eta
            loss = lmbd_elb * loss
            diffusion_loss_full_elbo = None
        
        elif self.diff_loss_type == "elbo_noise_scaling":
            lmbd_elb = 0.5 * torch.exp(-gamma*self.scalar) * d_gamma / eta
            loss = lmbd_elb * loss
            diffusion_loss_full_elbo = None

        elif self.diff_loss_type == "elbo_noise_scaling2":
            lmbd_elb = 0.5 * torch.exp(-gamma*0.5) * d_gamma / eta
            loss = lmbd_elb * loss
            diffusion_loss_full_elbo = None
        
        elif self.diff_loss_type == "x_0_prediction":
            # L_x weighting
            lmbd_x = 4 / (1 + eta) ** 2 
            
            loss = lmbd_x * loss 
            diffusion_loss_full_elbo = None
            #context_loss -> this lmdb_x implies we scaled full elbo by (lmbd_x/lmbd_elbo)
            #so to correctly scale the context loss we need to scale it by the same factor

            if loss.isnan().any():
                print("loss is nan")
                print("lmbd_x: ", lmbd_x)
                print("loss: ", loss)
                print("x: ", x)
                print("x_: ", x_)
                print("m: ", m)
                print("m_: ", m_)
                print("d_m: ", d_m)
                print("d_m_: ", d_m_)
                print("gamma: ", gamma)
                print("d_gamma: ", d_gamma)
                print("eta: ", eta)
                print("eps: ", eps)
                print("z: ", z)
                print("maybe this helps?")

            
        elif self.diff_loss_type == "half_elbo":
            lmbd_elb = 0.5 * torch.exp(-gamma) * d_gamma / eta
            lmbd_x = 4 / (1 + eta) ** 2 

            coef = (lmbd_x.detach()/lmbd_elb.detach())
            diffusion_loss_full_elbo = lmbd_elb * loss

            loss = coef * diffusion_loss_full_elbo
        elif self.diff_loss_type == "half_half_elbo":
            lmbd_elb = 0.5 * torch.exp(-gamma) * d_gamma / eta
            lmbd_x = 4 / (1 + eta) ** 2 

            coef = (lmbd_x/lmbd_elb.detach())
            diffusion_loss_full_elbo = lmbd_elb * loss

            loss = coef * diffusion_loss_full_elbo
        elif self.diff_loss_type == "two_losses":
            lmbd_elb = 0.5 * torch.exp(-gamma) * d_gamma / eta
            lmbd_x = 4 / (1 + eta) ** 2 

            pred_params = list(self.pred.parameters())
            noise_params = list(self.transform.parameters()) + list(self.gamma.parameters()) + list(self.vol_eta.parameters()) + list(self.context.parameters())
            for p in pred_params: p.requires_grad_(True)
            for p in noise_params: p.requires_grad_(False)
            loss_pred = lmbd_x * loss

            # 2) Compute noiser loss only
            for p in pred_params: p.requires_grad_(False)
            for p in noise_params: p.requires_grad_(True)
            loss_noise = lmbd_elb * loss

            # 3) Restore all requires_grad flags
            for p in pred_params + noise_params:
                p.requires_grad_(True)

            loss = loss_pred + loss_noise

            diffusion_loss_full_elbo = loss_noise

        # mask out the seq len dim (which is dim 1) where pad tokens are
        #loss = loss.masked_fill(pad_mask.unsqueeze(-1), 0)

        #loss = loss.sum(dim=1), dont reduce yet

        return loss, context_loss, diffusion_loss_full_elbo, x_, torch.exp(-gamma/2), d_gamma, loss_1

    def reconstruction_loss(self, x, t, embeddings):
        eps = torch.randn_like(embeddings)

        if (self.alpha_0 is not None) and (self.alpha_0.size(0) == t.size(0)):
            alpha = self.alpha_0
            sigma = self.sigma_0
        else:
            t = torch.zeros_like(t)
            print("recomputing reconstruction loss")
            if isinstance(self.context, NoneContext):
                gamma, _ = self.gamma(t)
            else:
                bs = t.size(0)
                context_hidden_dim = self.context.model.output_dim#context should always be input as [bs, hidden dim]
                context = torch.zeros(int(bs), int(context_hidden_dim), dtype = embeddings.dtype, device=embeddings.device)
                gamma = self.gamma.get_gamma(t, context) #prevent a tdir call -> noise boundaries are fixed at t=1 and t=0 and independent of context

            alpha = self.gamma.alpha_2(gamma) ** 0.5
            sigma = self.gamma.sigma_2(gamma) ** 0.5
            self.alpha_0 = alpha.detach()
            self.sigma_0 = sigma.detach()

        m, _ = self.transform.get_m_s(embeddings, t) #prevent another tdir call

        z_0 = alpha * m + sigma * eps
        
        decoder_nll = token_discrete_loss(z_0, self.pred.model.get_logits, x)
        
        return decoder_nll

    #this should only be computed if the forward process isnt implemented to have exactly n(0,1) at time 1. 
    def prior_loss(self, x, t):

        if (self.alpha_1 is not None) and (self.alpha_1.size(0) == t.size(0)):
            alpha = self.alpha_1
        else:
            t = torch.ones_like(t)
            print("recomputing prior loss")
            if isinstance(self.context, NoneContext):
                gamma, _ = self.gamma(t)
            else:
                bs = t.size(0)
                context_hidden_dim = self.context.model.output_dim#context should always be input as [bs, hidden dim]
                context = torch.zeros(int(bs), int(context_hidden_dim), dtype = x.dtype, device=x.device)
                gamma = self.gamma.get_gamma(t, context)

            alpha = self.gamma.alpha_2(gamma) ** 0.5
            self.alpha_1 = alpha.detach()

        m, _ = self.transform.get_m_s(x, t)
        mean = alpha * m     

        return mean ** 2

    
    def get_elbo_diffusion_loss(self, x, t):
        #this is not exactly the same as loss now, need to make sure to compute the full objective, but I did write it all out previously, but it is on a different piece of paper 
        with torch.set_default_dtype(torch.float64):
            x = self.pred.model.get_embeds(x).double()

            context, context_loss = self.context(x)
            context = context.double() if context is not None else None

            eps = torch.randn_like(x)

            if context is None:
                print("-0----------------------------------------shouldnt happen-----------------------------------------------------")
                gamma, d_gamma = self.gamma(t)
                gamma = gamma.double()
                d_gamma = d_gamma.double()
            else:
                gamma, d_gamma = self.gamma(t, context)
                gamma = gamma.double()
                d_gamma = d_gamma.double()

            alpha2 = self.gamma.alpha_2(gamma)
            sigma2 = self.gamma.sigma_2(gamma)

            alpha = alpha2 ** 0.5
            sigma = sigma2 ** 0.5

            (m, _), (d_m, _) = self.transform(x, t)

            eta = self.vol_eta(t).double()

            z = alpha * m + sigma * eps

            x_ = self.pred(z, t, context).double()

            (m_, _), (d_m_, _) = self.transform(x_, t)

            g2 = sigma2 * d_gamma * eta

            d_alpha = -0.5 * d_gamma * alpha * (1 - alpha2) 
            d_sigma = 0.5 * d_gamma * sigma * (1 - sigma2)  #TODO incorrect derivative 

            # compute backward flow '
            #epsilon = (z - alpha * m) / sigma #-> acutally epsilon should be exactly the same as eps
            #print(epsilon==eps, "epsilon == eps")
            #print(torch.all(m==x), "m == x")
            
            s = -eps / sigma#(alpha * m - z) / sigma2
            #f = d_alpha * m + d_sigma * eps
            f = d_alpha * m + alpha * d_m + d_sigma * eps
            f_B = f - (g2 / 2) * s

            #compute predicted backward flow 
            epsilon_ = (z - alpha * m_) / sigma
            #epsilon_ = eps
            s_ = (alpha * m_ - z) / sigma2
            #s_ = s
            #f_ = d_alpha * m_ + d_sigma * epsilon_
            f_ = d_alpha * m_ + alpha * d_m_ + d_sigma * epsilon_
            #f_ = f
            f_B_ = f_ - (g2 / 2) * s_
            #print(d_m_)

            #compute the loss
            elbo =(1/(2*g2)) * (f_B - f_B_) ** 2

            print(((m - m_)**2).mean(), "mean m")
            print(((s-s_)**2).mean(), "mean s")
            print(((f_B - f_B_)**2).mean(), "mean")
            #print((1/(2*g2)), "g2")

            return elbo, context_loss.sum(dim=-1)
    
    def get_elbo_reconstruction_loss(self, x, t):
        #this is also just cross entropy so it can call the same function as the reconstruction loss.
        embeddings = self.pred.model.get_embeds(x)

        eps = torch.randn_like(embeddings)

        t = torch.zeros_like(t)
           
        if isinstance(self.context, NoneContext):
            gamma, _ = self.gamma(t)
        else:
            bs = t.size(0)
            context_hidden_dim = self.context.model.output_dim#context should always be input as [bs, hidden dim]
            context = torch.zeros(int(bs), int(context_hidden_dim), dtype = embeddings.dtype, device=embeddings.device)
            gamma = self.gamma.get_gamma(t, context) #prevent a tdir call -> noise boundaries are fixed at t=1 and t=0 and independent of context

        alpha = self.gamma.alpha_2(gamma) ** 0.5
        sigma = self.gamma.sigma_2(gamma) ** 0.5

        m, _ = self.transform.get_m_s(embeddings, t) #prevent another tdir call

        z_0 = alpha * m + sigma * eps
        
        decoder_nll = token_discrete_loss(z_0, self.pred.model.get_logits, x, sum=True)
        
        return decoder_nll.mean(dim=-1)

    def get_elbo_prior_loss(self, x, t):
        embeddings = self.pred.model.get_embeds(x)

        B, D, H = embeddings.shape

        eps = torch.randn_like(embeddings)

        t = torch.ones_like(t)
           
        if isinstance(self.context, NoneContext):
            gamma, _ = self.gamma(t)
        else:
            bs = t.size(0)
            context_hidden_dim = self.context.model.output_dim#context should always be input as [bs, hidden dim]
            context = torch.zeros(int(bs), int(context_hidden_dim), dtype = embeddings.dtype, device=embeddings.device)
            gamma = self.gamma.get_gamma(t, context) #prevent a tdir call -> noise boundaries are fixed at t=1 and t=0 and independent of context

        alpha = self.gamma.alpha_2(gamma) ** 0.5
        sigma = self.gamma.sigma_2(gamma) ** 0.5

        m, _ = self.transform.get_m_s(embeddings, t) #prevent another tdir call

        f_m = alpha * m 
        f_s = sigma

        mu_flat = f_m.view(B, -1)                  # shape [B, D] where D=seq_len*hidden_dim

        # 2) Broadcast sigma up to mu1’s shape, then flatten
        #    torch.ones_like(mu1) has shape [B,seq_len,hidden_dim], so sigma * that
        sigma_bcast = f_s * torch.ones_like(f_m)  # broadcasts σ to [B,seq_len,hidden_dim]
        sigma_flat  = sigma_bcast.view(B, -1)      # now [B, D] exactly matching mu_flat

        # 3) Compute D (total latent dim)
        D = mu_flat.shape[1]

        # 4) Compute the three KL pieces
        term_trace  = torch.sum(sigma_flat**2,    dim=1)   # ∑ σ_i^2
        term_mean   = torch.sum(mu_flat**2,       dim=1)   # ∑ μ_i^2
        term_logdet = torch.sum(torch.log(sigma_flat**2), dim=1)  # ∑ log σ_i^2

        # 5) KL per example and mean over batch
        kl_per_example = 0.5*(term_trace + term_mean - D - term_logdet)
       
        return kl_per_example
        

