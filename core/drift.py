
import torch
from torch import Tensor


def compute_drift(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temp: float = 0.05,
) -> Tensor:
    """
    Compute the mean-shift drifting field V using optimized batched operations.
    
    Inspired by the MLX implementation, this uses a single concatenated target
    tensor for more efficient computation and better GPU utilization.

    Args:
        x:     Query points (generated samples) [N, D]
        y_pos: Positive samples (data) [N_pos, D]
        y_neg: Negative samples (generated) [N_neg, D]
        temp:  Kernel temperature

    Returns:
        V: Drift vectors [N, D]
    """
    N = x.shape[0]
    N_pos = y_pos.shape[0]
    
    # Concatenate targets for single batched distance computation
    targets = torch.cat([y_neg, y_pos], dim=0)  # [N_neg + N_pos, D]
    
    # Single batched pairwise distance computation [N, N_neg + N_pos]
    # More efficient than two separate cdist calls
    diff = x.unsqueeze(1) - targets.unsqueeze(0)  # [N, N_neg + N_pos, D]
    dist_sq = torch.sum(diff * diff, dim=-1)  # [N, N_neg + N_pos]
    dist = torch.sqrt(dist_sq + 1e-12)  # Add epsilon for numerical stability
    
    # Mask self-interactions (x ↔ y_neg diagonal)
    if N == y_neg.shape[0]:
        # Create mask for self-interactions in the y_neg part
        mask = torch.eye(N, device=x.device, dtype=torch.bool)
        # Pad mask to match full targets length
        mask = torch.cat([
            mask,
            torch.zeros(N, N_pos, device=x.device, dtype=torch.bool)
        ], dim=1)
        dist = torch.where(mask, torch.tensor(1e6, device=x.device), dist)
    
    # Compute kernel with temperature scaling
    kernel = torch.exp(-dist / temp)  # [N, N_neg + N_pos]
    
    # Doubly-normalized affinity (batch normalization along both dims)
    # Using geometric mean of row and column normalizations
    norm_row = kernel.sum(dim=-1, keepdim=True).clamp(min=1e-12)  # [N, 1]
    norm_col = kernel.sum(dim=-2, keepdim=True).clamp(min=1e-12)  # [1, N_neg + N_pos]
    normalizer = torch.sqrt(norm_row * norm_col)
    
    A = kernel / normalizer  # [N, N_neg + N_pos]
    
    # Split into positive and negative affinities
    A_neg = A[:, :N]       # [N, N_neg] (assumes y_neg = x)
    A_pos = A[:, N:]       # [N, N_pos]
    
    # Factorized weights for efficient computation
    # W_pos[i,j] = A_pos[i,j] * sum_k A_neg[i,k]
    # W_neg[i,k] = A_neg[i,k] * sum_j A_pos[i,j]
    sum_neg = A_neg.sum(dim=1, keepdim=True)  # [N, 1]
    sum_pos = A_pos.sum(dim=1, keepdim=True)  # [N, 1]
    
    W_pos = A_pos * sum_neg  # [N, N_pos]
    W_neg = A_neg * sum_pos  # [N, N_neg]
    
    # Compute drift vectors using factorized form
    # V[i] = sum_j sum_k A_pos[i,j] A_neg[i,k] (y_pos[j] - y_neg[k])
    V_pos = W_pos @ y_pos  # [N, D]
    V_neg = W_neg @ y_neg  # [N, D]
    V = V_pos - V_neg
    
    return V
