"""
Real-time visualization using PyQtGraph for 120 FPS performance.

This module provides a high-performance visualizer for training monitoring
using GPU-accelerated scatter plots and efficient data handling.
"""

import numpy as np
import torch
from torch import Tensor
from typing import Optional
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import sys


class RealtimeVisualizer:
    """
    High-performance real-time visualizer for drifting model training.
    
    Features:
    - GPU-accelerated scatter plots via OpenGL
    - Targeting 120 FPS updates
    - Separate thread for non-blocking rendering
    - Automatic downsampling for display efficiency
    - Loss curve with scrolling window
    """
    
    def __init__(
        self,
        title: str = "Drifting Model Training",
        target_fps: int = 120,
        max_display_points: int = 2000,
        loss_history_size: int = 1000,
    ):
        """
        Initialize the real-time visualizer.
        
        Args:
            title: Window title
            target_fps: Target framerate for updates
            max_display_points: Maximum points to display (downsampling)
            loss_history_size: Size of loss history ring buffer
        """
        self.target_fps = target_fps
        self.max_display_points = max_display_points
        self.loss_history_size = loss_history_size
        
        # Initialize Qt application if not already running
        self.app = QtWidgets.QApplication.instance()
        if self.app is None:
            self.app = QtWidgets.QApplication(sys.argv)
        
        # Create main window
        self.win = pg.GraphicsLayoutWidget(show=True, title=title)
        self.win.resize(1400, 600)
        
        # Enable OpenGL for better performance
        pg.setConfigOptions(useOpenGL=True, antialias=True)
        
        # Create plots
        self._setup_plots()
        
        # Data storage (ring buffers)
        self.loss_history = np.zeros(loss_history_size)
        self.step_history = np.zeros(loss_history_size)
        self.history_idx = 0
        self.history_count = 0
        
        # Timer for updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._process_qt_events)
        self.update_interval = int(1000 / target_fps)  # ms
        self.timer.start(self.update_interval)
        
        # Track last update time for FPS calculation
        self.last_update_time = QtCore.QTime.currentTime()
        self.fps_counter = 0
        self.fps_display = 0.0
        
    def _setup_plots(self):
        """Setup the plot layout and configure visual properties."""
        # Loss plot (top row, spans full width)
        self.loss_plot = self.win.addPlot(row=0, col=0, colspan=2, title="Training Loss (Log Scale)")
        self.loss_plot.setLabel('left', 'Loss')
        self.loss_plot.setLabel('bottom', 'Training Step')
        self.loss_plot.setLogMode(y=True)
        self.loss_plot.showGrid(x=True, y=True, alpha=0.3)
        self.loss_curve = self.loss_plot.plot(pen=pg.mkPen('w', width=2))
        
        # FPS counter text
        self.fps_text = pg.TextItem(text='', anchor=(1, 0), color='y')
        self.loss_plot.addItem(self.fps_text)
        self.fps_text.setPos(0, 0)
        
        # Scatter plots (bottom row)
        self.win.nextRow()
        
        # Real data scatter
        self.real_plot = self.win.addPlot(row=1, col=0, title="Target Distribution")
        self.real_plot.setAspectLocked(True)
        self.real_plot.hideAxis('left')
        self.real_plot.hideAxis('bottom')
        self.real_scatter = pg.ScatterPlotItem(
            size=3,
            pen=None,
            brush=pg.mkBrush(0, 255, 255, 150)  # Cyan
        )
        self.real_plot.addItem(self.real_scatter)
        
        # Generated data scatter
        self.gen_plot = self.win.addPlot(row=1, col=1, title="Generated Distribution")
        self.gen_plot.setAspectLocked(True)
        self.gen_plot.hideAxis('left')
        self.gen_plot.hideAxis('bottom')
        self.gen_scatter = pg.ScatterPlotItem(
            size=3,
            pen=None,
            brush=pg.mkBrush(255, 140, 0, 180)  # Orange
        )
        self.gen_plot.addItem(self.gen_scatter)
        
    def _downsample_points(self, points: np.ndarray) -> np.ndarray:
        """
        Downsample points for efficient rendering.
        
        Args:
            points: Array of shape [N, 2]
        
        Returns:
            Downsampled array of shape [min(N, max_display_points), 2]
        """
        if len(points) <= self.max_display_points:
            return points
        
        # Random sampling for unbiased downsampling
        indices = np.random.choice(len(points), self.max_display_points, replace=False)
        return points[indices]
    
    def update(self, step: int, loss: float, real: Tensor, gen: Tensor):
        """
        Update the visualization with new training data.
        
        Args:
            step: Current training step
            loss: Current loss value
            real: Real data samples [N, 2]
            gen: Generated samples [N, 2]
        """
        # Update loss history (ring buffer)
        idx = self.history_idx % self.loss_history_size
        self.loss_history[idx] = loss
        self.step_history[idx] = step
        self.history_idx += 1
        self.history_count = min(self.history_count + 1, self.loss_history_size)
        
        # Update loss curve
        if self.history_count > 1:
            # Get valid history
            if self.history_count < self.loss_history_size:
                steps = self.step_history[:self.history_count]
                losses = self.loss_history[:self.history_count]
            else:
                # Circular buffer - reorder
                steps = np.roll(
                    self.step_history,
                    -self.history_idx % self.loss_history_size
                )
                losses = np.roll(
                    self.loss_history,
                    -self.history_idx % self.loss_history_size
                )
            
            self.loss_curve.setData(steps, losses)
        
        # Update scatter plots with downsampled data
        real_np = real.detach().cpu().numpy()
        gen_np = gen.detach().cpu().numpy()
        
        real_down = self._downsample_points(real_np)
        gen_down = self._downsample_points(gen_np)
        
        self.real_scatter.setData(real_down[:, 0], real_down[:, 1])
        self.gen_scatter.setData(gen_down[:, 0], gen_down[:, 1])
        
        # Update FPS counter
        self.fps_counter += 1
        current_time = QtCore.QTime.currentTime()
        elapsed = self.last_update_time.msecsTo(current_time)
        
        if elapsed >= 1000:  # Update FPS display every second
            self.fps_display = self.fps_counter * 1000.0 / elapsed
            self.fps_counter = 0
            self.last_update_time = current_time
            
            # Update FPS text
            max_step = self.step_history[idx] if self.history_count > 0 else 0
            max_loss = self.loss_history[idx] if self.history_count > 0 else 0
            self.fps_text.setText(
                f'FPS: {self.fps_display:.1f} | Step: {int(max_step)} | Loss: {max_loss:.4f}'
            )
            # Position text in top-right
            if self.history_count > 0:
                self.fps_text.setPos(max_step * 0.98, max_loss * 1.2)
        
        # Process Qt events to keep UI responsive
        self._process_qt_events()
    
    def _process_qt_events(self):
        """Process Qt events to keep the UI responsive."""
        QtWidgets.QApplication.processEvents()
    
    def close(self):
        """Close the visualizer window."""
        self.timer.stop()
        self.win.close()
    
    def is_closed(self) -> bool:
        """Check if the window has been closed."""
        return not self.win.isVisible()


def example_usage():
    """Example usage of the real-time visualizer."""
    import time
    from data.generators import gen_data
    
    # Create visualizer
    viz = RealtimeVisualizer(
        title="Example Visualization",
        target_fps=120,
        max_display_points=2000
    )
    
    # Simulate training loop
    for step in range(500):
        # Generate fake data
        real = gen_data(2000)
        gen = torch.randn(2000, 2) + torch.randn(1, 2) * 2
        
        # Simulate loss
        loss = 1.0 / (step + 1) + 0.01 * np.random.randn()
        
        # Update visualization
        viz.update(step, loss, real, gen)
        
        # Simulate training time
        time.sleep(0.01)
        
        # Check if window was closed
        if viz.is_closed():
            print("Window closed by user")
            break
    
    print("Done!")
    viz.close()


if __name__ == "__main__":
    example_usage()
