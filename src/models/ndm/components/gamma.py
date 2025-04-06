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