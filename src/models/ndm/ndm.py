from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.distributions as D

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from src.models.ndm.components.gamma import Gamma
from src.models.ndm.components.transform import AffineTransform
from src.models.ndm.components.vol_eta import VolatilityEta
from src.models.ndm.components.predictor import Predictor

from their_utils.utils import token_discrete_loss
 


class NeuralDiffusion(nn.Module):
    def __init__(self, transform: AffineTransform, gamma: Gamma, vol_eta: VolatilityEta, pred: Predictor, diff_loss_type = "elbo", gamma_init=False, 
                 add_pure_x_pred=False):
        super().__init__()

        self.transform = transform
        self.gamma = gamma
        self.add_pure_x_pred = add_pure_x_pred
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

        diffusion_loss, diffusion_loss_full_elbo, embeddings_, _, _, _ = self.get_diffusion_loss(embeddings, t, pad_mask)

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
        
        return diffusion_loss, diffusion_loss_full_elbo, reconstruction_loss, prior_loss

            
        
    def get_diffusion_loss(self, x: Tensor, t: Tensor, pad_mask: Tensor):
        eps = torch.randn_like(x)

        gamma, d_gamma = self.gamma(t)
        alpha = self.gamma.alpha_2(gamma) ** 0.5
        sigma = self.gamma.sigma_2(gamma) ** 0.5

        (m, _), (d_m, _) = self.transform(x, t)

        eta = self.vol_eta(t)

        z = alpha * m + sigma * eps

        x_ = self.pred(z, t)

        (m_, _), (d_m_, _) = self.transform(x_, t)

        #so I would like to weight like x prediction so I should use this weighting instead
        #the l_x weighting above still includes the 1/2g(t) term but in my other formulation it actually doesnt, since g(t) >1? is that true, that could make the loss a bit lower
        #okay I take that back
        one_over_dgamma = torch.clamp(1 / (d_gamma), max=10000) #so it doesnt go to inf 

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
        
        elif self.diff_loss_type == "x_0_prediction":
            # L_x weighting
            lmbd_x = 4 / (1 + eta) ** 2 
            loss = lmbd_x * loss 
            diffusion_loss_full_elbo = None
            
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

        # mask out the seq len dim (which is dim 1) where pad tokens are
        #loss = loss.masked_fill(pad_mask.unsqueeze(-1), 0)

        #loss = loss.sum(dim=1), dont reduce yet

        return loss, diffusion_loss_full_elbo, x_, torch.exp(-gamma/2), d_gamma, loss_1

    def reconstruction_loss(self, x, t, embeddings):
        eps = torch.randn_like(embeddings)

        if (self.alpha_0 is not None) and (self.alpha_0.size(0) == t.size(0)):
            alpha = self.alpha_0
            sigma = self.sigma_0
        else:
            t = torch.zeros_like(t)
            print("recomputing reconstruction loss")
            gamma = self.gamma.get_gamma(t) #prevent a tdir call
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
            gamma = self.gamma.get_gamma(t) #prevent a tdir call
            alpha = self.gamma.alpha_2(gamma) ** 0.5
            self.alpha_1 = alpha.detach()

        m, _ = self.transform.get_m_s(x, t)
        mean = alpha * m     

        return mean ** 2
        

