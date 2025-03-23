import torch.nn as nn
from torch import Tensor

class Net(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, out_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        return self.net(t)