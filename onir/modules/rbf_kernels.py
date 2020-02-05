import torch
from torch import nn


class RbfKernelBank(nn.Module):
    def __init__(self, mus=None, sigmas=None, dim=0, requires_grad=True):
        super().__init__()
        self.mus = nn.Parameter(torch.tensor(mus), requires_grad=requires_grad)
        self.sigmas = nn.Parameter(torch.tensor(sigmas), requires_grad=requires_grad)
        self.dim = dim

    def forward(self, data):
        shape = list(data.shape)
        shape.insert(self.dim, 1)
        data = data.reshape(*shape)
        shape = [1]*len(data.shape)
        shape[self.dim] = -1
        mus, sigmas = self.mus.reshape(*shape), self.sigmas.reshape(*shape)
        adj = data - mus
        return torch.exp(-0.5 * adj * adj / sigmas / sigmas)

    def count(self):
        return self.mus.shape[0]

    @staticmethod
    def from_strs(mus='-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0', \
        sigmas='0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001', dim=-1, requires_grad=True):
        mus = [float(x) for x in mus.split(',')]
        sigmas = [float(x) for x in sigmas.split(',')]
        return RbfKernelBank(mus, sigmas, dim=dim, requires_grad=requires_grad)

    @staticmethod
    def evenly_spaced(count=11, sigma=0.1, rng=(-1, 1), dim=-1, requires_grad=True):
        mus = [x.item() for x in torch.linspace(rng[0], rng[1], steps=count)]
        sigmas = [sigma for _ in mus]
        return RbfKernelBank(mus, sigmas, dim=dim, requires_grad=requires_grad)
