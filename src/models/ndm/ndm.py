from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.distributions as D

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from improved_diffusion.ndm.components.gamma import Gamma
from improved_diffusion.ndm.components.transform import AffineTransform
from improved_diffusion.ndm.components.vol_eta import VolatilityEta
from improved_diffusion.ndm.components.predictor import Predictor
 


class NeuralDiffusion(nn.Module):
    def __init__(self, transform: AffineTransform, gamma: Gamma, vol_eta: VolatilityEta, pred: Predictor, mask_padding = False):
        super().__init__()

        self.transform = transform
        self.gamma = gamma
        self.vol_eta = vol_eta
        self.pred = pred

        self.mask_padding = mask_padding

        self.alpha_0 = None
        self.sigma_0 = None
        self.alpha_1 = None
        

    def forward(self, x: Tensor, t: Tensor, **model_kwargs):
        return self.pred(x, t, **model_kwargs)

    def get_losses(self, x: Tensor, t: Tensor,
                token_discrete_loss,
                compute_diffusion_loss=True,
                compute_prior_loss=False,
                compute_reconstruction_loss=True,
                reconstruction_loss_type = "collapse",
                compute_their_loss=False, #this wont be used here. 
                **model_kwargs):
        
        bs = x.size(0)
        pad_mask = x == 3
        embeddings = self.pred.model.get_embeds(x)    

        diffusion_loss = self.get_diffusion_loss(embeddings, t, pad_mask)

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

        # ELBO weighting
        lmbd_elb = 0.5 * torch.exp(-gamma) * d_gamma / eta

        # L_x weighting
        lmbd_x = 4 / (1 + eta) ** 2 #so I would like to weight like x prediction so I should use this weighting instead
        #the l_x weighting above still includes the 1/2g(t) term but in my other formulation it actually doesnt, since g(t) >1? is that true, that could make the loss a bit lower
        #okay I take that back

        loss = (1 + eta) / 2 * (m - m_) + 1 / d_gamma * (d_m - d_m_)
        loss = loss ** 2

        # Stabilises training
        #loss_x = (1 + eta) ** 2 / 4 * (x - x_) ** 2
        #loss = 0.5 * loss + 0.5 * loss_x

        coef = (lmbd_x.detach()/lmbd_elb.detach())

        loss = coef * (lmbd_elb * loss)
        # mask out the seq len dim (which is dim 1) where pad tokens are
        #loss = loss.masked_fill(pad_mask.unsqueeze(-1), 0)

        #loss = loss.sum(dim=1), dont reduce yet

        return loss

    def reconstruction_loss(self, x, t, embeddings, token_discrete_loss):
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
        

