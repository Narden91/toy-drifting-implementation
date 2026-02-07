# Toy Drifting Implementation

PyTorch implementation of generative modeling via drifting, based on the paper [Generative Modeling via Drifting](https://arxiv.org/abs/2602.04770).

## Features

- ⚡ **Optimized Performance**: 2-3x faster drift computation with batched operations
- 🎨 **Real-time Visualization**: 120 FPS capable PyQtGraph-based training monitor
- 📊 **Multiple Datasets**: 8-Gaussians and Checkerboard patterns
- 🔧 **Configurable**: YAML-based configuration system
- 📁 **Organized Outputs**: Automatic folder structure for results

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run training with default config (8-Gaussians dataset)
uv run python main.py

# Use checkerboard dataset
uv run python main.py --config config/checkerboard_config.yaml

# Run performance benchmark
python tests/benchmark_performance.py
```

## Performance

Optimized implementation with significant speedups:
- **Drift computation**: 1.6ms/iter (1.27M samples/sec on GPU)
- **Training iteration**: 4.4ms/iter
- **Memory usage**: ~301MB peak (batch size 2048)

See [OPTIMIZATION_GUIDE.md](doc/OPTIMIZATION_GUIDE.md) for details.

## Visualization

Two visualization modes:

1. **Real-time (PyQtGraph)**: 120 FPS interactive window during training
2. **Static (Matplotlib)**: High-quality exports saved to `outputs/`

Configure in YAML:
```yaml
visualization:
  enabled: true
  target_fps: 120
  update_every: 1
```

## Project Structure

```
toy-drifting-implementation/
├── main.py              # Entry point
├── train.py             # Training loop
├── config/              # Configuration files
├── core/                # Drift computation & loss
├── data/                # Data generators
├── models/              # Neural network architecture
├── utils/               # Visualization utilities
├── outputs/             # Generated images (auto-created)
│   ├── data/
│   ├── drift/
│   ├── comparison/
│   └── final/
└── tests/               # Benchmarks & tests
```

## Configuration

Edit `config/default.yaml`:
```yaml
training:
  batch_size: 2048
  lr: 0.001
  n_iter: 500

model:
  noise_dim: 32
  hidden_dim: 256
  temp: 0.05

data:
  dataset: "8gaussians"  # or "checkerboard"
```

## Citation

```bibtex
@article{drifting2025,
  title={Generative Modeling via Drifting},
  author={[Authors]},
  journal={arXiv preprint arXiv:2602.04770},
  year={2025}
}
```

## License

MIT
