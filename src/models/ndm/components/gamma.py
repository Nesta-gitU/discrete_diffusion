from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.distributions as D

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from src.models.nfdm.nfdm import t_dir
from math import prod
from functools import partial


class Gamma(nn.Module, ABC):
    @staticmethod
    def alpha_2(g):
        return torch.sigmoid(-g)

    @staticmethod
    def sigma_2(g):
        return torch.sigmoid(g)

    @abstractmethod
    def get_gamma(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(self, t: Tensor) -> tuple[Tensor, Tensor]:
        return t_dir(self.get_gamma, t)

class GammaLinear(Gamma):
    @staticmethod
    def get_gamma(t):
        return -10 + 20 * t

    @staticmethod
    def forward(t):
        g = GammaLinear.get_gamma(t)
        dg = torch.ones_like(t) * 20
        return g, dg
    
class GammaTheirs(Gamma):
    @staticmethod
    def get_gamma(t):
        def safe_logit(x, eps=1e-6):
            """
            Stable log( x / (1 - x) ) for x in (0, 1).

            Parameters
            ----------
            x   : torch.Tensor   -- input, any shape
            eps : float          -- clamp width; keeps gradients finite
            """
            x = x.clamp(eps, 1.0 - eps)            # avoid exactly 0 or 1
            return torch.log(x) - torch.log1p(-x)   # log(x) - log(1 - x)
        s = (0.99-t) * 0.0001
        sqrt = torch.sqrt(t+s)
        gamma = safe_logit(sqrt)

        #gamma = torch.log(1/(1-sqrt) - 1)
        #gamma = -10 + 20 * t
        return gamma


class PosLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.sp = nn.Softplus()

    def forward(self, x: Tensor) -> Tensor:
        weight = self.sp(self.weight)
        bias = self.bias
        return F.linear(x, weight, bias)


class GammaVDM(Gamma):
    def __init__(self, gamma_0=-10, gamma_1=10):
        super().__init__()

        self.gamma_0 = gamma_0
        self.gamma_1 = gamma_1
        self.fc1 = PosLinear(1, 1)
        self.fc2 = PosLinear(1, 1024)
        self.fc3 = PosLinear(1024, 1)

    def get_unnorm_gamma(self, x):
        x = self.fc1(x)

        y = self.fc2(x)
        y = torch.sigmoid(y)
        y = self.fc3(y)

        return x + y

    def get_gamma(self, t):
        x_0 = torch.zeros(1, 1,1).to(t.device)
        x_1 = torch.ones(1, 1,1).to(t.device)
        y_0 = torch.ones(1, 1,1) * (self.gamma_0)
        y_1 = torch.ones(1, 1,1) * self.gamma_1
        y_gap = y_1 - y_0

        y_0 = y_0.to(t.device)
        y_1 = y_1.to(t.device)

        x_adj = torch.cat([x_0, x_1, t], dim=0)
        y_adj = self.get_unnorm_gamma(x_adj)
        yo_0, yo_1, yo = y_adj[:1], y_adj[1:2], y_adj[2:]

        y = y_0 + (y_1 - y_0) * (yo - yo_0) / (yo_1 - yo_0)

        return y


class GammaAR(Gamma):
    def __init__(self, gamma_shape: int, gamma_min: float = -10, gamma_max: float = 10, learn_delta: bool = False):
        super().__init__()
        seq_len = gamma_shape[0]
        self.seq_len = gamma_shape[0]
        self.min_gamma = gamma_min
        self.max_minus_min_gamma = gamma_max - gamma_min
        
        # Sharp sigmoid window: duration = 1 / seq_len
        self.tau = 1 / (seq_len * 5)  # or 6, 8, etc.

        # Reversed offsets: last token starts first
        delta_vals = torch.linspace(1/seq_len, 1 - 1 / seq_len, seq_len)
        self.delta = nn.Parameter(delta_vals.flip(0).view(1, seq_len), requires_grad=learn_delta)

    def get_gamma(self, t):
        t = t.view(-1, 1)
        
        t_shifted = (t - self.delta) / self.tau  # [bs, seq_len]
        sigma = torch.sigmoid(t_shifted)
        gamma = self.min_gamma + self.max_minus_min_gamma * sigma
        gamma = gamma.view(-1, self.seq_len, 1)  # [bs, seq_len]
        return gamma

    def forward(self, t):
        t = t.view(-1, 1)  # [bs, 1]
        
        t_shifted = (t - self.delta) / self.tau  # [bs, seq_len]
        sigma = torch.sigmoid(t_shifted)

        gamma = self.min_gamma + self.max_minus_min_gamma * sigma
        dgamma_dt = (self.max_minus_min_gamma / self.tau) * sigma * (1 - sigma)

        gamma = gamma.view(-1, self.seq_len, 1)  # [bs, seq_len]
        dgamma_dt = dgamma_dt.view(-1, self.seq_len, 1)

        return gamma, dgamma_dt  # [bs, seq_len]


class GammaMuLAN(Gamma):
    #implement get_gamma and forward methods

    def __init__(self, gamma_shape, around_reference = False, gamma_min=-10, gamma_max=10):
        super().__init__()
        self.gamma_shape = gamma_shape
        self.n_features = prod(gamma_shape)
        self.min_gamma = gamma_min
        self.max_minus_min_gamma = gamma_max - gamma_min
        self.grad_min_epsilon = 0.001
        self.around_reference = around_reference

        #self.l1 = nn.Linear(self.n_features, self.n_features)
        #if I want the noise injection to not depend on the input, more like vdm, since input dependence will be injected through ndm function transformation, I can just make the input of
        #l1 to be 1. which means the input is just the time t. That is not fully equal to MuLAN since there the a, b, d are dependent only on input not on t.
        #that also mean I will still need to implement the get_gamma method, but not the forward method, we need tdir still. 
        self.l1 = nn.Linear(1, self.n_features)
        self.l2 = nn.Linear(self.n_features, self.n_features)
        self.l3_a = nn.Linear(self.n_features, self.n_features)
        self.l3_b = nn.Linear(self.n_features, self.n_features)
        self.l3_c = nn.Linear(self.n_features, self.n_features)

    def _eval_polynomial(self, a, b, c, t):
        # Polynomial evaluation
        polynomial = (
            (a ** 2) * (t ** 5) / 5.0
            + (b ** 2 + 2 * a * c) * (t ** 3) / 3.0
            + a * b * (t ** 4) / 2.0
            + b * c * (t ** 2)
            + (c ** 2 + self.grad_min_epsilon) * t)
        
        scale = ((a ** 2) / 5.0
                 + (b ** 2 + 2 * a * c) / 3.0
                 + a * b / 2.0
                 + b * c
                 + (c ** 2 + self.grad_min_epsilon))

        return self.min_gamma + self.max_minus_min_gamma * polynomial / scale
    
    def _grad_t(self, a, b, c, t):
        # derivative = (at^2 + bt + c)^2
        polynomial = (
        (a ** 2) * (t ** 4)
        + (b ** 2 + 2 * a * c) * (t ** 2)
        + a * b * (t ** 3) * 2.0
        + b * c * t * 2
        + (c ** 2))
        
        scale = ((a ** 2) / 5.0
                + (b ** 2 + 2 * a * c) / 3.0
                + a * b / 2.0
                + b * c
                + (c ** 2))

        return self.max_minus_min_gamma * polynomial / scale

    def _compute_coefficients(self, t):
        _h = torch.nn.functional.silu(self.l1(t))
        _h = torch.nn.functional.silu(self.l2(_h))
        a = self.l3_a(_h)
        b = self.l3_b(_h)
        c = 1e-3 + torch.nn.functional.softplus(self.l3_c(_h))
        return a, b, c

    def get_reference_gamma(self, t):
        def safe_logit(x, eps=1e-6):
            """
            Stable log( x / (1 - x) ) for x in (0, 1).

            Parameters
            ----------
            x   : torch.Tensor   -- input, any shape
            eps : float          -- clamp width; keeps gradients finite
            """
            x = x.clamp(eps, 1.0 - eps)            # avoid exactly 0 or 1
            return torch.log(x) - torch.log1p(-x)   # log(x) - log(1 - x)
        s = (0.99-t) * 0.0001
        sqrt = torch.sqrt(t+s)
        gamma = safe_logit(sqrt)

        #gamma = torch.log(1/(1-sqrt) - 1)
        #gamma = -10 + 20 * t
        return gamma
    
    def get_gamma(self, t):
        x = torch.ones_like(t)
        a, b, c = self._compute_coefficients(x)
        gamma = self._eval_polynomial(a, b, c, t)
        #print(gamma.shape, "before")

        #print(self.around_reference)
        if self.around_reference:
            #print("hallooooooo")
            reference_gamma = self.get_reference_gamma(t)
            #gamma_i = gamma + gamma’_i - gamma’
            #where gamma is the reference gamma, and
            #gamma’ = log D - log sum exp(-gamma’_i)
            gamma_mean = torch.log(torch.tensor(self.n_features)) - torch.logsumexp(-gamma, dim=-1)
            #print(reference_gamma.shape, "ref")
            #print(gamma_mean.shape, "mean")
            gamma = reference_gamma + gamma - gamma_mean.unsqueeze(-1) #should broadcast
            #print(gamma.shape, "after")

        #shape should be bs=t.shape[0], gamma_shape
        #how do I append a value to the shape though?
        gamma = gamma.view(-1, *self.gamma_shape)

        return gamma
    
    def forward(self, t):
        if self.around_reference:
            gamma, dgamma = t_dir(self.get_gamma, t)
            dgamma = torch.clamp(dgamma, min=self.grad_min_epsilon)
            return gamma, dgamma

        x = torch.ones_like(t)
        a, b, c = self._compute_coefficients(x)
        dg = self._grad_t(a, b, c, t)
        dg = dg.clamp(min=self.grad_min_epsilon)
        dg = dg.view(-1, *self.gamma_shape)
        return self.get_gamma(t), dg

class GammaMuLANContext(Gamma):
    #implement get_gamma and forward methods

    def __init__(self, gamma_shape, around_reference = False, gamma_min=-10, gamma_max=10):
        super().__init__()
        self.gamma_shape = gamma_shape
        self.n_features = prod(gamma_shape)
        self.min_gamma = gamma_min
        self.max_minus_min_gamma = gamma_max - gamma_min
        self.grad_min_epsilon = 0.001
        self.around_reference = around_reference

        #self.l1 = nn.Linear(self.n_features, self.n_features)
        #if I want the noise injection to not depend on the input, more like vdm, since input dependence will be injected through ndm function transformation, I can just make the input of
        #l1 to be 1. which means the input is just the time t. That is not fully equal to MuLAN since there the a, b, d are dependent only on input not on t.
        #that also mean I will still need to implement the get_gamma method, but not the forward method, we need tdir still. 
        self.l1 = nn.Linear(self.n_features * 128, self.n_features) #need to down prod first
        #self.l1 = nn.Linear(1, self.n_features)
        self.l2 = nn.Linear(self.n_features, self.n_features)
        self.l3_a = nn.Linear(self.n_features, self.n_features)
        self.l3_b = nn.Linear(self.n_features, self.n_features)
        self.l3_c = nn.Linear(self.n_features, self.n_features)

    def _eval_polynomial(self, a, b, c, t):
        # Polynomial evaluation
        polynomial = (
            (a ** 2) * (t ** 5) / 5.0
            + (b ** 2 + 2 * a * c) * (t ** 3) / 3.0
            + a * b * (t ** 4) / 2.0
            + b * c * (t ** 2)
            + (c ** 2 + self.grad_min_epsilon) * t)
        
        scale = ((a ** 2) / 5.0
                 + (b ** 2 + 2 * a * c) / 3.0
                 + a * b / 2.0
                 + b * c
                 + (c ** 2 + self.grad_min_epsilon))

        return self.min_gamma + self.max_minus_min_gamma * polynomial / scale
    
    def _grad_t(self, a, b, c, t):
        # derivative = (at^2 + bt + c)^2
        polynomial = (
        (a ** 2) * (t ** 4)
        + (b ** 2 + 2 * a * c) * (t ** 2)
        + a * b * (t ** 3) * 2.0
        + b * c * t * 2
        + (c ** 2))
        
        scale = ((a ** 2) / 5.0
                + (b ** 2 + 2 * a * c) / 3.0
                + a * b / 2.0
                + b * c
                + (c ** 2)) #this is one of the differences 

        return self.max_minus_min_gamma * polynomial / scale

    def _compute_coefficients(self, x):
        x = x.flatten(start_dim=1)
        #print(x.shape, "x shape, after flatten")
        #x = torch.ones_like(x)
        print(x.shape, "x shape, after ones")
        _h = torch.nn.functional.silu(self.l1(x))
        _h = torch.nn.functional.silu(self.l2(_h))
        a = self.l3_a(_h)
        b = self.l3_b(_h)
        c = 1e-3 + torch.nn.functional.softplus(self.l3_c(_h))
        #print(a.shape, "a shape")
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        c = c.unsqueeze(-1)
        print(a.shape, "a shape, after unsqueeze")
        return a, b, c

    def get_reference_gamma(self, t):
        def safe_logit(x, eps=1e-6):
            """
            Stable log( x / (1 - x) ) for x in (0, 1).

            Parameters
            ----------
            x   : torch.Tensor   -- input, any shape
            eps : float          -- clamp width; keeps gradients finite
            """
            x = x.clamp(eps, 1.0 - eps)            # avoid exactly 0 or 1
            return torch.log(x) - torch.log1p(-x)   # log(x) - log(1 - x)
        s = (0.99-t) * 0.0001
        sqrt = torch.sqrt(t+s)
        gamma = safe_logit(sqrt)

        #gamma = torch.log(1/(1-sqrt) - 1)
        #gamma = -10 + 20 * t
        return gamma
    
    def get_gamma(self, t, x):
        #print(x.shape, "x shape") -> [bs, seqlen, n_features]
        a, b, c = self._compute_coefficients(x)
        gamma = self._eval_polynomial(a, b, c, t)
        #print(gamma.shape, "before")

        #print(self.around_reference)
        if self.around_reference:
            #print("hallooooooo")
            reference_gamma = self.get_reference_gamma(t)
            #gamma_i = gamma + gamma’_i - gamma’
            #where gamma is the reference gamma, and
            #gamma’ = log D - log sum exp(-gamma’_i)
            gamma_mean = torch.log(torch.tensor(self.n_features)) - torch.logsumexp(-gamma, dim=-1)
            #print(reference_gamma.shape, "ref")
            #print(gamma_mean.shape, "mean")
            gamma = reference_gamma + gamma - gamma_mean.unsqueeze(-1) #should broadcast
            #print(gamma.shape, "after")

        #shape should be bs=t.shape[0], gamma_shape
        #how do I append a value to the shape though?
        #print(gamma.shape, "before")
        gamma = gamma.view(x.shape[0], *self.gamma_shape)
        #print(gamma.shape, "gamma shape")
        #print(t.shape)
        return gamma
    
    def forward(self, t, x):
        if self.around_reference:
            partial_gamma = partial(self.get_gamma, x=x)
            gamma, dgamma = t_dir(partial_gamma, t)
            dgamma = torch.clamp(dgamma, min=self.grad_min_epsilon)
            return gamma, dgamma

        a, b, c = self._compute_coefficients(x)
        dg = self._grad_t(a, b, c, t)
        dg = dg.clamp(min=self.grad_min_epsilon)
        dg = dg.view(x.shape[0], *self.gamma_shape)
        return self.get_gamma(t, x), dg


class GammaMuLANtDir(GammaMuLAN):

    #implement get_gamma and forward methods

    def __init__(self, gamma_shape, gamma_min=-10, gamma_max=10):
        super().__init__(gamma_shape, gamma_min, gamma_max)

    def get_gamma(self, t):
        a, b, c = self._compute_coefficients(t)
        gamma_flat = self._eval_polynomial(a, b, c, t)
        #shape should be bs=t.shape[0], gamma_shape
        #how do I append a value to the shape though?
        
        gamma = gamma_flat.view(-1, *self.gamma_shape)
        #print(gamma.shape, "gamma shape")
        return gamma
        
    def forward(self, t):
        gamma, dgamma = t_dir(self.get_gamma, t)
        dgamma = torch.clamp(dgamma, min=self.grad_min_epsilon)
        return gamma, dgamma