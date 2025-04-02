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

class AffineTransform(nn.Module, ABC):
    @abstractmethod
    def get_m_s(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        raise NotImplementedError

    def forward(self, x: Tensor, t: Tensor) -> tuple[tuple[Tensor, Tensor], tuple[Tensor, Tensor]]:
        def f(t_in):
            return self.get_m_s(x, t_in)

        return t_dir(f, t)


class AffineTransformID(AffineTransform):
    @staticmethod
    def get_m_s(x, t):
        m = x
        s = torch.ones_like(x)
        return m, s

    @staticmethod
    def forward(x, t):
        m, s = AffineTransformID.get_m_s(x, t)

        dm = torch.zeros_like(x)
        ds = torch.zeros_like(x)

        return (m, s), (dm, ds)


class AffineTransformHalfNeural(AffineTransform):
    def __init__(self, model):
        super().__init__()

        self.net = model

    def get_m_s(self, x, t):
        
        m = self.net(x, t.squeeze(-1).squeeze(-1))

        m = x + t * m
        s = torch.ones_like(x)

        return m, s

