from abc import ABC, abstractmethod
from typing import Callable, Optional

import numpy as np

import torch
from torch import nn, Tensor
from torch.nn import functional as F
import torch.distributions as D

import matplotlib.pyplot as plt
from tqdm import tqdm, trange

from src.models.ndm.nets.mlp import Net

class VolatilityEta(nn.Module, ABC):
    @abstractmethod
    def forward(self, t: Tensor) -> Tensor:
        raise NotImplementedError

class VolatilityEtaOne(nn.Module):
    def forward(self, t):
        return torch.ones_like(t)


class VolatilityEtaNeural(nn.Module, ABC):
    def __init__(self):
        super().__init__()

        self.net = Net(1, 1)
        self.sp = nn.Softplus()

    def forward(self, t):
        return self.sp(self.net(t))