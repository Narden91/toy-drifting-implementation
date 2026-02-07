
import torch
from torch import Tensor
from .drift import compute_drift

def drifting_loss(gen: Tensor, pos: Tensor, temp: float = 0.05) -> Tensor:
    """
    Compute per-sample drifting loss.

    Args:
        gen: Generated samples [N, D] (with gradient)
        pos: Data (positive) samples [N_pos, D]
        temp: Kernel temperature

    Returns:
        Per-sample loss [N] (equals ||V(x)||^2 per sample)
    """
    with torch.no_grad():
        V = compute_drift(gen, pos, gen, temp=temp)
        target = (gen + V).detach()
    return ((gen - target) ** 2).sum(dim=-1)
