import torch
import torch.nn as nn

class Correlation(nn.Module):
# calculates the correlation loss
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def _rho(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.flatten(1)                     # (B,N)
        y = y.flatten(1)
        c  = (x * y).sum(-1)
        n1 = x.pow(2).sum(-1)
        n2 = y.pow(2).sum(-1)
        return c / (torch.sqrt(n1 * n2) + self.eps)   # (B,)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        ρ = self._rho(x, y)
        return ρ.mean()
