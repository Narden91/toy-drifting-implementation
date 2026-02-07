
import torch
from torch import Tensor
from .drift import compute_drift

def drifting_loss(gen: Tensor, pos: Tensor, temp: float = 0.05) -> Tensor:
    """
    Compute per-sample drifting loss (Algorithm 1 from paper).
    
    L = E[ || f(eps) - stopgrad(f(eps) + V(f(eps))) ||^2 ]
    
    The loss value equals ||V||^2. Gradients flow only through the prediction
    f(eps), not through the frozen target f(eps) + V.

    Args:
        gen: Generated samples [N, D] (with gradient)
        pos: Data (positive) samples [N_pos, D]
        temp: Kernel temperature

    Returns:
        Per-sample loss [N] (equals ||V(x)||^2 per sample)
    """
    # Compute drift field (V is fixed target direction)
    with torch.no_grad():
        V = compute_drift(gen, pos, gen, temp=temp)
    
    # Target is current position + drift
    target = (gen + V).detach()
    
    # Loss pulls gen towards target
    return ((gen - target) ** 2).sum(dim=-1)
