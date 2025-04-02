import torch
from torch import nn, Tensor
import torch.distributions as D
from abc import ABC, abstractmethod
from typing import Optional

class ContSampler(nn.Module):
    def __init__(self):
        super().__init__()
        

    def forward(self, bs, device):
        return self.sample(bs,device)

    def sample(self, batch_size, device):
        t = torch.rand(batch_size, 1).to(device)
        return t, None

class TimeSampler(nn.Module, ABC):
    def __init__(self, salt_fraction: Optional[int] = None):
        super().__init__()

        self._salt_fraction = salt_fraction

    @abstractmethod
    def prob(self, t: Tensor) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample(self, bs: int) -> Tensor:
        raise NotImplementedError

    def loss(self, loss: Tensor, t: Tensor) -> Tensor:
        """
        In terms of minimization of the variance, this loss is not quite correct. Firstly, in lit module,
        we detach t and loss. Theoretically we should differentiate end-to-end through loss to obtain
        the true gradient w.r.t. parameters of the proposal distribution. However, to do this, we must
        differentiate through the training step second time just to optimize the proposal distribution,
        which is too expensive. Therefore, we detach t and loss and work with biased gradient. Secondly,
        we should take into account the salting, which we don't.
        """
        self.device = t.device
        p = self.prob(t)

        l2 = loss ** 2
        p2 = p ** 2

        return l2 / p2

    def forward(self, bs: int, device) -> tuple[Tensor, Tensor]:
        self.device = device
        

        t = self.sample(bs).to(device)
        dtype = t.dtype

        if self._salt_fraction is not None:
            assert bs % self._salt_fraction == 0

            bs2 = bs // self._salt_fraction
            bs1 = bs - bs2

            un = D.Uniform(
                torch.tensor([0.], dtype=dtype, device=self.device),
                torch.tensor([1.], dtype=dtype, device=self.device)
            )
            u = un.sample(torch.Size((bs2,)))

            t = torch.cat([t[:bs1], u], dim=0)

            p = self.prob(t)

            k = 1 / self._salt_fraction
            p = p * (1 - k) + k
        else:
            p = self.prob(t)

        return t, p


class UniformSampler(TimeSampler):
    def __init__(self, salt_fraction: Optional[int] = None):
        super().__init__(salt_fraction)

        self.register_buffer("_l", torch.tensor(0.))
        self.register_buffer("_r", torch.tensor(1.))

    @property
    def _u(self) -> D.Uniform:
        return D.Uniform(self._l, self._r)

    def prob(self, t: Tensor) -> Tensor:
        return self._u.log_prob(t).squeeze(dim=1).exp()

    def sample(self, bs: int) -> Tensor:
        return self._u.sample(torch.Size((bs, 1)))


class BucketSampler(TimeSampler):
    def __init__(self, n: int = 100, salt_fraction: Optional[int] = None):
        super().__init__(salt_fraction)

        self._logits = nn.Parameter(torch.ones(n))

    @property
    @abstractmethod
    def _bucket_prob(self) -> Tensor:
        raise NotImplementedError

    @property
    @abstractmethod
    def _bucket_width(self) -> Tensor:
        raise NotImplementedError

    @property
    def _bucket_height(self) -> Tensor:
        return self._bucket_prob / self._bucket_width

    @property
    def _bucket_bounds(self) -> tuple[Tensor, Tensor]:
        w = self._bucket_width

        dtype = w.dtype
        device = self.device

        b_r = torch.cumsum(w, dim=0)
        b_l = torch.cat([torch.zeros(1, dtype=dtype, device=self.device), b_r[:-1]])

        return b_l, b_r

    def prob(self, t: Tensor) -> Tensor:
        t = t.flatten()

        t, ids_t = torch.sort(t)
        inv_ids_t = torch.argsort(ids_t)

        b_l, _ = self._bucket_bounds

        ids_p = torch.searchsorted(b_l, t, right=True) - 1

        p = self._bucket_height
        p = torch.index_select(p, 0, ids_p)
        p = torch.index_select(p, 0, inv_ids_t)

        return p

    def sample(self, bs: int) -> Tensor:
        b_p = self._bucket_prob
        b_l, b_r = self._bucket_bounds

        dtype = b_p.dtype
        #device = b_p.device

        cat = D.Categorical(b_p)
        ids = cat.sample(torch.Size((bs,))).to(self.device)

        un = D.Uniform(
            torch.tensor(0., dtype=dtype, device=self.device),
            torch.tensor(1., dtype=dtype, device=self.device)
        )
        u = un.sample(torch.Size((bs,))).to(self.device)

        t = torch.index_select(b_l, 0, ids) + torch.index_select(b_r - b_l, 0, ids) * u
        t = t[:, None]

        return t


class UniformBucketSampler(BucketSampler):
    @property
    def _bucket_prob(self) -> Tensor:
        return torch.softmax(self._logits, dim=0)

    @property
    def _bucket_width(self) -> Tensor:
        logits = self._logits.to(self.device)
        dtype = logits.dtype
        #device = logits.device
        n = logits.shape[0]
        return torch.ones(n, dtype=dtype, device=self.device) / n