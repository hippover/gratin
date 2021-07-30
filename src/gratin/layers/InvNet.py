import torch
import torch.nn as nn
import numpy as np
from .diverse import MLP


class ACB(nn.Module):
    def __init__(self, dim_theta, dim_mu, stable_s=False, alpha=1.7):
        """
        dim_theta : dimension of the parameters vector
        dim_mu : dimension of the summary vector
        """
        super(ACB, self).__init__()

        self.n_1 = int(dim_theta // 2)
        self.n_2 = int(np.ceil(dim_theta / 2))
        assert self.n_1 + self.n_2 == dim_theta

        self.permutation = nn.Parameter(torch.randperm(dim_theta), requires_grad=False)
        self.inv_permutation = nn.Parameter(
            torch.argsort(self.permutation), requires_grad=False
        )

        self.s_1 = MLP(
            [
                self.n_2 + dim_mu,
                (self.n_2 + dim_mu) * 4,
                self.n_2 + dim_mu + 32,
                self.n_1,
            ]
        )
        self.t_1 = MLP(
            [
                self.n_2 + dim_mu,
                (self.n_2 + dim_mu) * 4,
                self.n_2 + dim_mu + 32,
                self.n_1,
            ]
        )
        self.s_2 = MLP(
            [
                self.n_1 + dim_mu,
                (self.n_1 + dim_mu) * 4,
                self.n_1 + dim_mu + 32,
                self.n_2,
            ]
        )
        self.t_2 = MLP(
            [
                self.n_1 + dim_mu,
                (self.n_1 + dim_mu) * 4,
                self.n_1 + dim_mu + 32,
                self.n_2,
            ]
        )
        self.stable_s = stable_s

        self.alpha = alpha

    def clamp(self, s):
        s = (2.0 * self.alpha / np.pi) * torch.atan(s / self.alpha)
        return s

    def forward(self, theta, x, inverse=False, log_det_J=False):

        if not inverse:
            u = torch.index_select(theta, 1, self.permutation)

            u_1 = torch.index_select(u, 1, torch.arange(self.n_1, device=u.device))
            u_2 = torch.index_select(
                u, 1, torch.arange(self.n_2, device=u.device) + self.n_1
            )

            U_2 = torch.cat((u_2, x), dim=1)
            s_1 = self.clamp(self.s_1(U_2))
            if self.stable_s:
                s_1 = s_1 - torch.mean(s_1, dim=1, keepdim=True)
            v_1 = u_1 * torch.exp(s_1) + self.t_1(U_2)

            V_1 = torch.cat((v_1, x), dim=1)
            s_2 = self.clamp(self.s_2(V_1))
            if self.stable_s:
                s_2 = s_2 - torch.mean(s_2, dim=1, keepdim=True)
            v_2 = u_2 * torch.exp(s_2) + self.t_2(V_1)

            result = torch.cat((v_1, v_2), dim=1)
            log_det_J_loss = torch.sum(s_1, dim=1) + torch.sum(s_2, dim=1)

            if log_det_J:
                return result, log_det_J_loss
            else:
                return result

        else:
            v = theta
            v_1 = torch.index_select(v, 1, torch.arange(self.n_1, device=v.device))
            v_2 = torch.index_select(
                v, 1, torch.arange(self.n_2, device=v.device) + self.n_1
            )

            V_1 = torch.cat((v_1, x), dim=1)
            u_2 = (v_2 - self.t_2(V_1)) * torch.exp(-self.clamp(self.s_2(V_1)))

            U_2 = torch.cat((u_2, x), dim=1)
            u_1 = (v_1 - self.t_1(U_2)) * torch.exp(-self.clamp(self.s_1(U_2)))

            inv = torch.cat((u_1, u_2), dim=1)

            result = torch.index_select(inv, 1, self.inv_permutation)

            return result


class InvertibleNet(nn.Module):
    def __init__(self, dim_theta, dim_x, n_blocks, stable_s=False, alpha_clip=1.7):
        super(InvertibleNet, self).__init__()
        self.ACBs = nn.Sequential(
            *[
                ACB(dim_theta, dim_x, stable_s=stable_s, alpha=alpha_clip)
                for k in range(n_blocks)
            ]
        )

    def forward(self, theta, x, inverse=False, return_intermediates=False):
        if not inverse:
            log_J_total = 0.0
            z_intermediates = [theta]
            for acb in self.ACBs:
                theta, log_J = acb(theta, x, inverse=False, log_det_J=True)
                z_intermediates.append(theta.clone())
                log_J_total = log_J_total + log_J
            z = theta
            if not return_intermediates:
                return z, log_J_total
            else:
                return z, log_J_total, z_intermediates
        else:
            theta_intermediates = [theta]
            z = theta
            for acb in self.ACBs[::-1]:
                z = acb(z, x, inverse=True, log_det_J=False)
                theta_intermediates.append(z.clone())
            theta = z
            if not return_intermediates:
                return theta
            else:
                return theta, theta_intermediates