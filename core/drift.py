
import torch
from torch import Tensor


def compute_drift(
    x: Tensor,
    y_pos: Tensor,
    y_neg: Tensor,
    temp: float = 0.05,
) -> Tensor:
    """
    Compute the mean-shift drifting field V (Algorithm 2 from paper).
    
    Mean-shift drifting field (compact form):
      V(x) = (1 / (Z_p Z_q)) E_{y+~p, y-~q}[ k(x,y+) k(x,y-) (y+ - y-) ]
    
    Implementation uses doubly-normalized softmax (over both x and y axes)
    and factorized weight computation.

    Args:
        x:     Query points (generated samples) [N, D]
        y_pos: Positive samples (data) [N_pos, D]
        y_neg: Negative samples (generated) [N_neg, D]
        temp:  Kernel temperature

    Returns:
        V: Drift vectors [N, D]
    """
    N, N_pos = x.shape[0], y_pos.shape[0]
    
    # Pairwise L2 distances
    dist_pos = torch.cdist(x, y_pos)  # [N, N_pos]
    dist_neg = torch.cdist(x, y_neg)  # [N, N_neg]
    
    # Mask self-interactions (y_neg = x in standard usage)
    if N == y_neg.shape[0]:
        dist_neg = dist_neg + torch.eye(N, device=x.device) * 1e4
    
    # Logits: -dist / temp
    logit = torch.cat([
        -dist_pos / temp,
        -dist_neg / temp,
    ], dim=1)  # [N, N_pos + N_neg]
    
    # Doubly-normalized affinity: softmax over both dims, geometric mean
    A_row = logit.softmax(dim=-1)  # normalize over y  [N, N_pos + N_neg]
    A_col = logit.softmax(dim=-2)  # normalize over x  [N, N_pos + N_neg]
    A = (A_row * A_col).sqrt()     # geometric mean
    
    # Split into positive and negative affinities
    A_pos = A[:, :N_pos]   # [N, N_pos]
    A_neg = A[:, N_pos:]   # [N, N_neg]
    
    # Factorized weights for compact form:
    # W_pos[i,j] = A_pos[i,j] * sum_k A_neg[i,k]
    # W_neg[i,k] = A_neg[i,k] * sum_j A_pos[i,j]
    W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
    W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)
    
    # V[i] = sum_j sum_k A_pos[i,j] A_neg[i,k] (y_pos[j] - y_neg[k])
    V = W_pos @ y_pos - W_neg @ y_neg
    
    return V
