# Drifting Models: A New Paradigm for One-Step Generative Modeling

> **Paper**: "Generative Modeling via Drifting" by Deng et al. (ICML 2026, MIT & Harvard)
> 
> **Key Innovation**: Train a generator that evolves its output distribution *during training*, enabling **one-step inference** (1-NFE) without iterative denoising.

---

## Executive Summary

Drifting Models offer a fundamentally different approach to generative modeling compared to diffusion/flow-based methods:

| Aspect | Diffusion/Flow Models | Drifting Models |
|--------|----------------------|-----------------|
| **When distribution evolves** | At inference (multi-step) | At training |
| **Inference cost** | N iterations (expensive) | 1 forward pass (cheap) |
| **Core mechanism** | Numerical ODE/SDE solving | Drifting field V → 0 at equilibrium |
| **State-of-the-art** | FID ~1.4-2.0 (multi-step) | FID 1.54 (1-NFE) on ImageNet 256×256 |

---

## The Core Idea: Training-Time Distribution Evolution

### The Problem with Traditional Generative Models

Traditional diffusion/flow models do iterative refinement **at inference**:

```mermaid
flowchart LR
    subgraph Inference["Inference Time (Expensive)"]
        N1[Noise] --> |"step 1"| X1[x₁]
        X1 --> |"step 2"| X2[x₂]
        X2 --> |"..."| XN[xₙ]
        XN --> |"step N"| I[Image]
    end
    style Inference fill:#ffe6e6,stroke:#ff6666
```

### Drifting Models: Evolve During Training

Drifting models move the "iteration" to **training time**:

```mermaid
flowchart LR
    subgraph Training["Training Time (Evolution)"]
        direction TB
        F1["f₁ → q₁"] --> |"optimizer update"| F2["f₂ → q₂"]
        F2 --> |"optimizer update"| F3["f₃ → q₃"]
        F3 --> |"..."| FN["fₙ → qₙ ≈ p"]
    end
    
    subgraph Inference["Inference Time (1 forward pass)"]
        Noise["ε ~ N(0,I)"] --> |"f(ε)"| Sample["x = f(ε) ~ q ≈ p"]
    end
    
    Training --> |"trained model"| Inference
    
    style Training fill:#e6f3ff,stroke:#6699ff
    style Inference fill:#e6ffe6,stroke:#66ff66
```

---

## Key Concepts Explained

### 1. Pushforward Distribution

The **pushforward** is simply "what happens to a distribution when you pass it through a function":

```mermaid
flowchart LR
    P["Prior p_ε<br/>(Gaussian noise)"] --> |"f (neural net)"| Q["Pushforward q = f#p_ε<br/>(generated samples)"]
    
    T["Target p_data<br/>(real data)"] -.-> |"want q ≈ p_data"| Q
    
    style P fill:#ffcccc
    style Q fill:#ffffcc
    style T fill:#ccffcc
```

**Goal**: Find f such that **q = f#pε ≈ pdata**

### 2. The Drifting Field V

The drifting field tells each sample **where to move** to better match the data distribution:

```mermaid
flowchart TB
    subgraph DriftingField["Drifting Field V(x)"]
        X["Sample x (generated)"] --> V["V(x) = attraction - repulsion"]
        
        subgraph Pos["Attraction V⁺"]
            P1["y⁺ ~ p_data"] --> |"pull toward data"| V
        end
        
        subgraph Neg["Repulsion V⁻"]
            N1["y⁻ ~ q (generated)"] --> |"push away from crowd"| V
        end
    end
    
    style Pos fill:#ccffcc,stroke:#66cc66
    style Neg fill:#ffcccc,stroke:#cc6666
```

**Intuition**: 
- **Attraction** (V⁺): Move toward nearby data points
- **Repulsion** (V⁻): Spread out from other generated samples
- At **equilibrium** (q = p): forces balance → **V = 0**

### 3. The Anti-Symmetry Property

The key mathematical insight: if the drifting field is **anti-symmetric**, then distributions matching implies zero drift:

$$V_{p,q}(x) = -V_{q,p}(x) \quad \forall x$$

$$\Rightarrow \quad q = p \implies V_{p,q}(x) = 0 \quad \forall x$$

**Proof**: If q = p, then V_{p,q} = V_{q,p} = -V_{p,q} → V_{p,q} = 0

---

## The Drifting Field Formula

### Mean-Shift Formulation

The drifting field uses kernel-weighted mean-shift:

```mermaid
flowchart TB
    subgraph Formula["V(x) = V⁺(x) - V⁻(x)"]
        subgraph VPlus["V⁺ (attraction)"]
            VP["V⁺(x) = (1/Z_p) E_p[k(x,y⁺)(y⁺ - x)]"]
        end
        subgraph VMinus["V⁻ (repulsion)"]
            VM["V⁻(x) = (1/Z_q) E_q[k(x,y⁻)(y⁻ - x)]"]
        end
    end
    
    subgraph Kernel["Kernel k(x,y)"]
        K["k(x,y) = exp(-||x-y|| / τ)<br/>τ = temperature"]
    end
```

### Compact Form (Used in Implementation)

Combining the two terms:

$$V_{p,q}(x) = \frac{1}{Z_p Z_q} \mathbb{E}_{y^+ \sim p, y^- \sim q}\left[ k(x, y^+) \cdot k(x, y^-) \cdot (y^+ - y^-) \right]$$

This is **anti-symmetric** because swapping y⁺ ↔ y⁻ flips the sign of (y⁺ - y⁻).

---

## The Training Loss

### Fixed-Point Formulation

At equilibrium, samples should stay put: **x = x + V(x)** when V = 0.

Training uses a **stop-gradient** trick (similar to self-supervised learning):

$$\mathcal{L} = \mathbb{E}_\epsilon\left[ \left\| f_\theta(\epsilon) - \text{stopgrad}\left(f_\theta(\epsilon) + V(f_\theta(\epsilon))\right) \right\|^2 \right]$$

```mermaid
flowchart LR
    subgraph Loss["Training Loss"]
        E["ε ~ noise"] --> F["f_θ(ε) = x<br/>(prediction)"]
        F --> |"compute V"| V["x + V(x)"]
        V --> |"stopgrad"| T["target (frozen)"]
        F --> |"MSE"| L["Loss = ||x - target||²"]
        T --> L
    end
    
    style F fill:#ffffcc
    style T fill:#ccccff
```

**Key insight**: The loss value equals **||V||²**. Minimizing loss → V → 0 → distributions match!

---

## Algorithm Overview

```mermaid
flowchart TB
    subgraph Training["Training Loop"]
        direction TB
        S1["1. Sample noise ε ~ p_ε"] --> S2["2. Generate x = f_θ(ε)"]
        S2 --> S3["3. Sample data y⁺ ~ p_data"]
        S3 --> S4["4. Use x as negatives (y⁻ = x)"]
        S4 --> S5["5. Compute V = compute_drift(x, y⁺, y⁻)"]
        S5 --> S6["6. Build target = stopgrad(x + V)"]
        S6 --> S7["7. Loss = MSE(x, target)"]
        S7 --> S8["8. Backprop & update θ"]
        S8 --> S1
    end
    
    subgraph Inference["Inference (1-NFE)"]
        I1["Sample ε ~ N(0,I)"] --> I2["Return f_θ(ε)"]
    end
    
    style Training fill:#e6f3ff,stroke:#333
    style Inference fill:#e6ffe6,stroke:#333
```

---

## Implementation Details

### Doubly-Normalized Affinity Matrix

The paper uses a clever normalization scheme for computing V efficiently:

```mermaid
flowchart TB
    subgraph Compute["Computing V"]
        D1["Compute distances:<br/>dist_pos = cdist(x, y⁺)<br/>dist_neg = cdist(x, y⁻)"]
        D1 --> L["Logits = -dist / τ"]
        L --> S1["Softmax over y (row-wise)"]
        L --> S2["Softmax over x (col-wise)"]
        S1 --> A["A = √(A_row × A_col)<br/>(geometric mean)"]
        S2 --> A
        A --> W["Factorized weights:<br/>W_pos = A_pos × Σ A_neg<br/>W_neg = A_neg × Σ A_pos"]
        W --> V["V = W_pos @ y_pos - W_neg @ y_neg"]
    end
```

### Why Doubly-Normalized?

Single softmax normalizes only over samples. Double normalization:
- Stabilizes training across different batch sizes
- Improves gradient flow
- Maintains anti-symmetry property

---

## Feature Space Drifting

For high-dimensional data (images), drifting is computed in a **feature space**:

```mermaid
flowchart LR
    subgraph FeatureSpace["Feature Space Drifting"]
        X["x = f_θ(ε)<br/>(image)"] --> Phi["φ(x)<br/>(features)"]
        Y["y ~ p_data<br/>(real image)"] --> PhiY["φ(y)<br/>(features)"]
        Phi --> V["V in feature space"]
        PhiY --> V
    end
    
    subgraph Encoder["Feature Encoder φ"]
        E1["ResNet (MoCo, SimCLR)"]
        E2["MAE (latent space)"]
    end
```

**Key finding**: Self-supervised encoders (MoCo, SimCLR, MAE) work best because they place semantically similar samples close together—exactly what the kernel needs!

---

## Classifier-Free Guidance (CFG)

Drifting models support CFG **at training time** (not inference):

$$\tilde{q}(\cdot|c) = (1-\gamma) q_\theta(\cdot|c) + \gamma \cdot p_{data}(\cdot|\emptyset)$$

This means adding some unconditional data samples as extra negatives during training.

```mermaid
flowchart TB
    subgraph CFG["CFG in Drifting Models"]
        P["Positives: y⁺ ~ p_data(·|c)"]
        N1["Negatives (generated): y⁻ ~ q_θ(·|c)"]
        N2["Negatives (uncond data): y⁻ ~ p_data(·|∅)"]
        N1 --> Mix["Mix with rate γ"]
        N2 --> Mix
        Mix --> V["Compute V"]
        P --> V
    end
```

**Result**: The generator learns to produce samples that are "more class-conditional" while still doing **1-NFE inference**.

---

## Results Summary

### ImageNet 256×256 (Latent Space)

| Method | NFE | FID ↓ |
|--------|-----|-------|
| DiT-XL/2 | 500 | 2.27 |
| SiT-XL/2 | 500 | 2.06 |
| iMeanFlow-XL/2 | 1 | 1.72 |
| **Drifting Model L/2** | **1** | **1.54** |

### ImageNet 256×256 (Pixel Space)

| Method | NFE | FID ↓ |
|--------|-----|-------|
| StyleGAN-XL | 1 | 2.30 |
| GigaGAN | 1 | 3.45 |
| **Drifting Model L/16** | **1** | **1.61** |

---

## Key Takeaways

1. **Paradigm shift**: Move iteration from inference to training
2. **One-step generation**: No ODE/SDE solving needed at inference
3. **Anti-symmetry**: Ensures V → 0 when distributions match
4. **Feature space**: Self-supervised encoders provide the right similarity metric
5. **State-of-the-art**: Beats multi-step methods with single forward pass

---

## Mathematical Notation Summary

| Symbol | Meaning |
|--------|---------|
| f | Generator network (noise → samples) |
| ε ~ pε | Input noise (typically Gaussian) |
| x = f(ε) | Generated sample |
| q = f#pε | Pushforward distribution (generated) |
| p = pdata | Target data distribution |
| V(x) | Drifting field (tells sample where to move) |
| k(x,y) | Kernel function (similarity measure) |
| τ | Temperature parameter |
| y⁺ ~ p | Positive samples (data) |
| y⁻ ~ q | Negative samples (generated) |
