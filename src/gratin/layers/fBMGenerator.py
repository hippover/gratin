import torch
import torch.nn as nn
import numpy as np


class fBMGenerator(nn.Module):
    def __init__(self, dim, **kwargs):
        self.dim = dim
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

    def get_dx_cov(self, alpha, T, tau=np.inf, lam=1.0):

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

        C = self.f(k, alpha.view(-1, 1, 1).double(), tau.view(-1, 1, 1), lam=lam)
        # min_EV = np.min(np.linalg.eigvals(C))
        # assert min_EV > 0.0, "min EV %.2f, alpha = %.2f, tau = %d" % (min_EV, alpha, tau)

        return C.double()

    def forward(self, alpha, tau, diffusion, T):
        # alpha : array de taille (BS)
        # tau : array de taille (BS)
        # T : int

        BS = alpha.shape[0]
        C = self.get_dx_cov(alpha, T=T, tau=tau, lam=1.0)
        try:
            # Returns a lower triangular matrix
            L = torch.cholesky(C).float()
        except Exception as e:
            print("alpha =")
            print(alpha)
            print("tau = ")
            print(tau)
            print("T = ")
            print(T)
            print(e)
            raise

        du = torch.randn(
            (BS, T - 1, self.dim), device=alpha.device, requires_grad=False
        )
        du = du * (torch.sqrt(diffusion).view(BS, 1, 1).expand(du.size()))

        dx = L @ du
        return torch.cat(
            [
                torch.zeros((BS, 1, self.dim), device=alpha.device),
                torch.torch.cumsum(dx, dim=1),
            ],
            dim=1,
        )
