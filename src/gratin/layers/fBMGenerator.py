import torch
import torch.nn as nn
import numpy as np


class fBMGenerator(nn.Module):
    def __init__(self, T, dim, **kwargs):
        self.dim = dim
        self.T = T
        super(fBMGenerator, self).__init__(**kwargs)

    def f_(self, DT, alpha):
        return 0.5 * (
            torch.pow(torch.abs(DT + 1), alpha)
            + torch.pow(torch.abs(DT - 1), alpha)
            - 2 * torch.pow(torch.abs(DT), alpha)
        )

    def f(self, DT, alpha, tau, lam=1):
        R = self.f_(DT, alpha)
        R = torch.where(
            torch.abs(DT) >= tau, R * torch.exp(-lam * (torch.abs(DT) - tau)), R
        )
        return R

    def get_dx_cov(self, alpha, T, tau=np.inf, lam=1):

        BS = alpha.shape[0]
        t = torch.arange(T - 1, device=alpha.device, requires_grad=False).view(1, -1)
        # t : (1,T-1)
        t = t.expand(BS, -1)
        # t : (BS,T-1)
        t = t.view(BS, T - 1, 1)
        # t: (BS,T-1,1)
        i = t
        j = torch.transpose(t, 1, 2)
        k = j - i

        C = self.f(k, alpha.view(-1, 1, 1), tau.view(-1, 1, 1), lam=lam)
        # min_EV = np.min(np.linalg.eigvals(C))
        # assert min_EV > 0.0, "min EV %.2f, alpha = %.2f, tau = %d" % (min_EV, alpha, tau)

        return C

    def forward(self, alpha, tau):
        # alpha : array de taille (BS)
        # tau : array de taille (BS)
        # T : int
        BS = alpha.shape[0]

        C = self.get_dx_cov(alpha, T=self.T, tau=tau, lam=1.0)
        L = torch.cholesky(C)

        du = torch.randn(
            (BS, self.T - 1, self.dim), device=alpha.device, requires_grad=False
        )

        dx = L @ du
        return torch.cat(
            [
                torch.zeros((BS, 1, self.dim), device=alpha.device),
                torch.torch.cumsum(dx, dim=1),
            ],
            dim=1,
        )
