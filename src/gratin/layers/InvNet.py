import torch
import torch.nn as nn
import numpy as np
from .diverse import MLP


class ACB(nn.Module):
    def __init__(self, dim_theta, dim_mu, stable_s=False, alpha=1.7, s_abs=4, t_abs=4):
        """
        dim_theta : dimension of the parameters vector
        dim_mu : dimension of the summary vector
        """
        super(ACB, self).__init__()

        self.n_1 = int(dim_theta // 2)
        self.n_2 = int(np.ceil(dim_theta / 2))
        assert self.n_1 + self.n_2 == dim_theta

        self.s_limit = 0.0

        weight_norm = True  # "Density estimation using real NVP" paper uses this
        residual = False
        s_params = {
            "residual": residual,
            "use_weight_norm": weight_norm,
            "out_range": (-s_abs, s_abs),
            "activation": "ELU",
            "use_batch_norm": False,
        }
        t_params = {
            "residual": residual,
            "use_weight_norm": weight_norm,
            "out_range": (-t_abs, t_abs),
            "activation": "ELU",
            "use_batch_norm": False,
        }

        self.permutation = nn.Parameter(torch.randperm(dim_theta), requires_grad=False)
        self.inv_permutation = nn.Parameter(
            torch.argsort(self.permutation), requires_grad=False
        )

        self.s_1 = MLP(
            [
                self.n_2 + dim_mu,
                64,
                64,
                64,
                self.n_1,
            ],
            **s_params
        )
        self.t_1 = MLP(
            [
                self.n_2 + dim_mu,
                64,
                64,
                64,
                self.n_1,
            ],
            **t_params
        )
        self.s_2 = MLP(
            [
                self.n_1 + dim_mu,
                64,
                64,
                64,
                self.n_2,
            ],
            **s_params
        )
        self.t_2 = MLP(
            [
                self.n_1 + dim_mu,
                64,
                64,
                64,
                self.n_2,
            ],
            **t_params
        )
        self.stable_s = stable_s

        self.alpha = alpha

    def clamp(self, s):
        return (2.0 * self.alpha / np.pi) * torch.atan(s / self.alpha)

    def clamp_loss(self, s):
        #
        #mean_before = torch.mean(s)
        s = torch.clamp(s, max=self.s_limit, min=-np.inf)
        #mean_after = torch.mean(s)
        #print(
        #    "s_lim = %.3f\t Mean before %.2f \t after %.2f"
        #    % (self.s_limit, mean_before.item(), mean_after.item())
        #)
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
            v_1 = (u_1 * torch.exp(s_1)) + self.t_1(U_2)

            V_1 = torch.cat((v_1, x), dim=1)
            s_2 = self.clamp(self.s_2(V_1))
            if self.stable_s:
                s_2 = s_2 - torch.mean(s_2, dim=1, keepdim=True)
            v_2 = (u_2 * torch.exp(s_2)) + self.t_2(V_1)

            result = torch.cat((v_1, v_2), dim=1)
            log_det_J_loss = torch.sum(self.clamp_loss(s_1), dim=1) + torch.sum(
                self.clamp_loss(s_2), dim=1
            )

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
    def __init__(
        self,
        dim_theta,
        dim_x,
        n_blocks,
        stable_s=False,
        alpha_clip=1.7,
        t_abs=4,
        s_abs=4,
    ):
        super(InvertibleNet, self).__init__()
        self.ACBs = nn.Sequential(
            *[
                ACB(
                    dim_theta,
                    dim_x,
                    stable_s=stable_s,
                    alpha=alpha_clip,
                    t_abs=t_abs,
                    s_abs=s_abs,
                )
                for k in range(n_blocks)
            ]
        )

    def set_s_lim(self, s_lim):
        for acb in self.ACBs:
            acb.s_limit = s_lim

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
