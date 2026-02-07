"""
Performance benchmark script for drifting model optimizations.

This script measures the impact of optimizations by comparing:
- Drift computation throughput (samples/sec)
- Memory usage (MB)
- Training iteration time (ms/iter)
"""

import time
import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.drift import compute_drift
from data.generators import gen_data, get_device
from models.drifting_model import DriftingModel


def benchmark_drift_computation(n_samples: int = 2048, n_iterations: int = 100, device: torch.device = None):
    """Benchmark drift field computation performance."""
    if device is None:
        device = get_device()
    
    print(f"\n=== Drift Computation Benchmark ===")
    print(f"Device: {device}")
    print(f"Samples: {n_samples}, Iterations: {n_iterations}")
    
    # Generate test data
    x = torch.randn(n_samples, 2, device=device)
    y_pos = torch.randn(n_samples, 2, device=device)
    y_neg = x  # Standard usage
    
    # Warmup
    for _ in range(10):
        _ = compute_drift(x, y_pos, y_neg, temp=0.05)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        V = compute_drift(x, y_pos, y_neg, temp=0.05)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start_time
    avg_time = elapsed / n_iterations * 1000  # ms
    throughput = n_samples * n_iterations / elapsed
    
    print(f"Average time: {avg_time:.3f} ms/iter")
    print(f"Throughput: {throughput:.0f} samples/sec")
    
    return avg_time, throughput


def benchmark_memory_usage(batch_size: int = 2048, device: torch.device = None):
    """Benchmark memory usage during training."""
    if device is None:
        device = get_device()
    
    print(f"\n=== Memory Usage Benchmark ===")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    model = DriftingModel(noise_dim=32, hidden_dim=256, temp=0.05).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Run a few iterations
    for _ in range(10):
        pos = gen_data(batch_size, device=device)
        loss = model(pos, n_gen=batch_size).mean()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        print(f"Peak memory: {peak_memory:.2f} MB")
        return peak_memory
    else:
        print("Memory profiling only available on CUDA devices")
        return None


def benchmark_training_iteration(batch_size: int = 2048, n_iterations: int = 50, device: torch.device = None):
    """Benchmark full training iteration time."""
    if device is None:
        device = get_device()
    
    print(f"\n=== Training Iteration Benchmark ===")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}, Iterations: {n_iterations}")
    
    model = DriftingModel(noise_dim=32, hidden_dim=256, temp=0.05).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Warmup
    for _ in range(10):
        pos = gen_data(batch_size, device=device)
        loss = model(pos, n_gen=batch_size).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.perf_counter()
    for _ in range(n_iterations):
        pos = gen_data(batch_size, device=device)
        loss = model(pos, n_gen=batch_size).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start_time
    avg_time = elapsed / n_iterations * 1000  # ms
    
    print(f"Average time: {avg_time:.3f} ms/iter")
    print(f"Throughput: {batch_size * n_iterations / elapsed:.0f} samples/sec")
    
    return avg_time


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("Drifting Model Performance Benchmarks")
    print("=" * 60)
    
    device = get_device()
    
    # Run benchmarks
    drift_time, drift_throughput = benchmark_drift_computation(
        n_samples=2048, n_iterations=100, device=device
    )
    
    memory_usage = benchmark_memory_usage(batch_size=2048, device=device)
    
    train_time = benchmark_training_iteration(
        batch_size=2048, n_iterations=50, device=device
    )
    
    # Summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"Drift Computation: {drift_time:.3f} ms/iter ({drift_throughput:.0f} samples/sec)")
    if memory_usage:
        print(f"Peak Memory: {memory_usage:.2f} MB")
    print(f"Training Iteration: {train_time:.3f} ms/iter")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
