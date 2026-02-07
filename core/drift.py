
import torch
from torch import Tensor

def compute_drift(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temp: float = 0.05,
) -> Tensor:
    """
    Compute the mean-shift drifting field V.

    Args:
        x:     Query points (generated samples) [N, D]
        y_pos: Positive samples (data) [N_pos, D]
        y_neg: Negative samples (generated) [N_neg, D]
        temp:  Kernel temperature

    Returns:
        V: Drift vectors [N, D]
    """
    N, N_pos = x.shape[0], y_pos.shape[0]

    # pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos)                   # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)                    # [N, N_neg]

    # mask self-interactions (y_neg = x in standard usage)
    if N == y_neg.shape[0]:
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e6

    # logits: -dist / temp
    logit = torch.cat([
        -dist_pos / temp,
        -dist_neg / temp,
    ], dim=1)                                           # [N, N_pos + N_neg]

    # doubly-normalized affinity: softmax over both dims, geometric mean
    # We use stable softmax (subtract max) typically, but PyTorch softmax handles it reasonable well.
    # For very small temp, we might have numerical issues.
    
    A_row = logit.softmax(dim=-1)                       # normalize over y
    A_col = logit.softmax(dim=-2)                       # normalize over x
    A = (A_row * A_col).sqrt()

    # split into positive and negative affinities
    A_pos = A[:, :N_pos]                                # [N, N_pos]
    A_neg = A[:, N_pos:]                                # [N, N_neg]

    # factorized weights for compact form:
    # W_pos[i,j] = A_pos[i,j] * sum_k A_neg[i,k]
    # W_neg[i,k] = A_neg[i,k] * sum_j A_pos[i,j]
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

    # V[i] = sum_j sum_k A_pos[i,j] A_neg[i,k] (y_pos[j] - y_neg[k])
    V = W_pos @ y_pos - W_neg @ y_neg

    return V
