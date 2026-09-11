import math

import torch
from torch.distributions import Beta

def l2_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-7) -> torch.Tensor:


    return x / torch.clamp(x.norm(dim=dim, keepdim=True), min=eps)


class PowerSphericalDistribution:
    def __init__(self, mu: torch.Tensor, kappa: torch.Tensor, eps: float = 1e-7):
        self.eps = eps
        self.mu = l2_norm(x = mu, eps = eps)  # [..., m]


        self.kappa = torch.clamp(kappa, min=0.0)

        self.m = self.mu.shape[-1]
        self.d = self.m - 1
        beta_const = 0.5 * self.d
        self.alpha = self.kappa + beta_const  # [...,]
        self.beta = torch.as_tensor(
            beta_const, dtype=self.kappa.dtype, device=self.kappa.device
        ).expand_as(self.kappa)

    def _log_normalizer(self) -> torch.Tensor:
        # log N_X(κ,d) = -[ (α+β)log 2 + β log π + lgamma(α) - lgamma(α+β) ]
        return (
            -(self.alpha + self.beta) * math.log(2.0)
            - self.beta * math.log(math.pi)
            - torch.lgamma(self.alpha)
            + torch.lgamma(self.alpha + self.beta)
        )

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        dot = (self.mu * x).sum(dim=-1).clamp(-1.0, 1.0)
        return self._log_normalizer() + self.kappa * torch.log1p(dot)

    def entropy(self) -> torch.Tensor:
        # H = -[ log N_X + κ ( log 2 + ψ(α) - ψ(α+β) ) ]
        return -(
            self._log_normalizer()
            + self.kappa
            * (
                math.log(2.0)
                + (torch.digamma(self.alpha) - torch.digamma(self.alpha + self.beta))
            )
        )

    def kl_to_uniform(self) -> torch.Tensor:
        # KL(q || U(S^{d})) = -H(q) + log |S^{d}|
        d = torch.as_tensor(self.d, dtype=self.kappa.dtype, device=self.kappa.device)
        log_area = (
            math.log(2.0)
            + 0.5 * (d + 1.0) * math.log(math.pi)
            - torch.lgamma(0.5 * (d + 1.0))
        )
        return -self.entropy() + log_area

    def rsample(self):
        
        if torch.isnan(self.kappa).any() or torch.isnan(self.alpha).any():
            print("\n" + "="*50)
            print("🚨 致命警告: Beta 分布采样前检测到 NaN!")
            # 打印非 nan 部分的最大值，看看是不是数值爆炸了
            valid_kappa = self.kappa[~torch.isnan(self.kappa)]
            if len(valid_kappa) > 0:
                print(f"   当前有效的 Kappa 最大值: {valid_kappa.max().item():.4f}")
            else:
                print("   Kappa 已经全部变成 NaN 了！")
            print("="*50 + "\n")
        # ==========================================
        
        Z = Beta(self.alpha, self.beta).rsample()  # [*S, *B]
        t = (2.0 * Z - 1.0).unsqueeze(-1)  # [*S, *B, 1]

        # 2) v ~ U(S^{m-2})
        v = torch.randn(
            *self.mu.shape[:-1],
            self.m - 1,
            device=self.mu.device,
            dtype=self.mu.dtype,
        )  # [*S, *B, m-1]
        v = l2_norm(v, eps = self.eps)

        y = torch.cat(
            [t, torch.sqrt(torch.clamp(1 - t**2, min=0.0)) * v], dim=-1
        )  # [*S, *B, m]

        e1 = torch.zeros_like(self.mu)
        e1[..., 0] = 1.0
        u = l2_norm(e1 - self.mu, eps = self.eps)
        if u.dim() < y.dim():
            u = u.view((1,) * (y.dim() - u.dim()) + u.shape)
        z = y - 2.0 * (y * u).sum(dim=-1, keepdim=True) * u

        parallel = (self.mu - e1).abs().sum(dim=-1, keepdim=True) < 1e-6
        if parallel.any():
            p = parallel
            if p.dim() < y.dim() - 1:
                p = p.view((1,) * (y.dim() - 1 - p.dim()) + p.shape)
            z = torch.where(p, y, z)
        return z
