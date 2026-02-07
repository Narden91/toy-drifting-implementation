# Implementation Analysis: main.py vs Paper

This document analyzes whether `main.py` correctly implements the Drifting Models paper.

---

## ✅ Summary: The implementation is CORRECT

The `main.py` file faithfully implements the core algorithms from the paper for 2D toy data, with all key mathematical concepts properly translated to code.

---

## Detailed Comparison

### 1. Drifting Field V (Algorithm 2)

#### Paper Formula (Equation 11)
$$V_{p,q}(x) = \frac{1}{Z_p Z_q} \mathbb{E}_{y^+ \sim p, y^- \sim q}\left[ k(x, y^+) k(x, y^-) (y^+ - y^-) \right]$$

#### Implementation (`compute_drift` function, lines 107-159)

| Paper Specification | Implementation | Status |
|---------------------|----------------|--------|
| Kernel k(x,y) = exp(-‖x-y‖/τ) | `logit = -dist / temp` → softmax | ✅ Correct |
| Pairwise distances | `torch.cdist(x, y_pos)` and `cdist(x, y_neg)` | ✅ Correct |
| Mask self-interactions | `dist_neg += eye(N) * 1e6` when y_neg = x | ✅ Correct |
| Double normalization | `A_row.softmax(dim=-1)`, `A_col.softmax(dim=-2)`, `A = sqrt(A_row * A_col)` | ✅ Correct |
| Factorized weights | `W_pos = A_pos * A_neg.sum(...)` | ✅ Correct |
| Final drift | `V = W_pos @ y_pos - W_neg @ y_neg` | ✅ Correct |

**Code excerpt**:
```python
# doubly-normalized affinity: softmax over both dims, geometric mean
A_row = logit.softmax(dim=-1)  # normalize over y
A_col = logit.softmax(dim=-2)  # normalize over x
A = (A_row * A_col).sqrt()

# factorized weights
W_pos = A_pos * A_neg.sum(dim=1, keepdim=True)
W_neg = A_neg * A_pos.sum(dim=1, keepdim=True)

V = W_pos @ y_pos - W_neg @ y_neg
```

This exactly matches Algorithm 2 from the paper (Appendix A.1).

---

### 2. Training Loss (Algorithm 1)

#### Paper Formula (Equation 6)
$$\mathcal{L} = \mathbb{E}_\epsilon\left[ \left\| f_\theta(\epsilon) - \text{stopgrad}\left(f_\theta(\epsilon) + V\right) \right\|^2 \right]$$

#### Implementation (`drifting_loss` function, lines 171-186)

| Paper Specification | Implementation | Status |
|---------------------|----------------|--------|
| Stop-gradient on target | `with torch.no_grad(): V = compute_drift(...)` and `target = (gen + V).detach()` | ✅ Correct |
| MSE loss | `((gen - target) ** 2).sum(dim=-1)` | ✅ Correct |
| y_neg = generated samples | `compute_drift(gen, pos, gen, ...)` | ✅ Correct |

**Code excerpt**:
```python
def drifting_loss(gen: Tensor, pos: Tensor, temp: float = 0.05) -> Tensor:
    with torch.no_grad():
        V = compute_drift(gen, pos, gen, temp=temp)
        target = (gen + V).detach()
    return ((gen - target) ** 2).sum(dim=-1)
```

This exactly matches Algorithm 1 from the paper.

---

### 3. Generator Architecture

#### Paper Specification
- MLP noise → samples for 2D toy data
- DiT-like transformer for ImageNet (not applicable to toy implementation)

#### Implementation (`Net` class, lines 72-91)

| Paper Specification | Implementation | Status |
|---------------------|----------------|--------|
| Input: noise ε | `noise_dim=32` latent input | ✅ Correct |
| Output: samples x | `out_dim=2` for 2D | ✅ Correct |
| Hidden layers | 4 hidden layers, 256 units | ✅ Appropriate for toy |
| Activations | SELU | ✅ Reasonable choice |

---

### 4. DriftingModel Class (lines 192-228)

| Paper Specification | Implementation | Status |
|---------------------|----------------|--------|
| One-step generator | `generate(n)` → single forward pass | ✅ Correct |
| Training with drifting loss | `forward(pos, n_gen)` computes loss | ✅ Correct |
| Temperature parameter | `self.temp = temp` (default 0.05) | ✅ Correct |

---

### 5. Training Procedure (lines 309-348)

| Paper Specification | Implementation | Status |
|---------------------|----------------|--------|
| Sample noise ε | Implicit in `DriftingModel.forward()` | ✅ Correct |
| Sample data y⁺ | `pos = data_fn(batch_size)` | ✅ Correct |
| Use generated as negatives | Automatic in `drifting_loss` | ✅ Correct |
| Optimizer | Adam (paper uses AdamW for ImageNet) | ✅ Appropriate for toy |

---

## Minor Differences (Not Issues)

### 1. Temperature Values
- **Paper**: Uses multiple temperatures {0.02, 0.05, 0.2} and sums losses
- **Implementation**: Uses single temperature (0.05)
- **Verdict**: ✅ Acceptable for toy demonstration

### 2. Batch Structure
- **Paper (ImageNet)**: Complex batching with Nc class labels, Npos, Nneg
- **Implementation**: Simple batching (all samples same "class")
- **Verdict**: ✅ Correct for unconditioned 2D data

### 3. Feature Encoder
- **Paper (ImageNet)**: Uses ResNet/MAE feature extractor φ
- **Implementation**: No feature encoder (direct 2D space)
- **Verdict**: ✅ Correct—toy 2D data doesn't need feature extraction

### 4. CFG (Classifier-Free Guidance)
- **Paper**: Supports CFG conditioning
- **Implementation**: Not implemented
- **Verdict**: ✅ Not needed for unconditional 2D toy data

---

## Summary Table

| Component | Paper | main.py | Match |
|-----------|-------|---------|-------|
| Drifting field V formula | Eq. 11 | `compute_drift()` | ✅ |
| Kernel k(x,y) | Eq. 12 | Softmax on -dist/τ | ✅ |
| Double normalization | Alg. 2 | √(A_row × A_col) | ✅ |
| Factorized weights | Alg. 2 | W_pos, W_neg | ✅ |
| Stop-gradient loss | Eq. 6 / Alg. 1 | `drifting_loss()` | ✅ |
| One-step inference | 1-NFE | `generate()` | ✅ |
| Self as negatives | Alg. 1 | Passed `gen` as `y_neg` | ✅ |
| Mask self in dist | Alg. 2 | `+eye(N)*1e6` | ✅ |

---

## Conclusion

**The implementation is a faithful and correct translation of the paper's core algorithms.**

It correctly implements:
1. ✅ The mean-shift drifting field with kernel weighting
2. ✅ The doubly-normalized affinity matrix
3. ✅ The factorized weight computation
4. ✅ The stop-gradient training objective
5. ✅ One-step generation at inference

The simplifications made (single temperature, no feature encoder, no CFG) are all appropriate for a 2D toy demonstration and do not affect the correctness of the core algorithm.
