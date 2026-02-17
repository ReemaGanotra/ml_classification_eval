"""
Shared utility functions for ML/CV assignment.
Handles reproducibility, logging, and common operations.
"""

import os
import random
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
SEED = 42
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Create directories
for dir_path in [DATA_DIR, MODELS_DIR, ARTIFACTS_DIR]:
    dir_path.mkdir(exist_ok=True)


def set_seed(seed: int = SEED):
    """
    Set random seeds for reproducibility across all libraries.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Set deterministic operations for TensorFlow
    tf.config.experimental.enable_op_determinism()
    
    logging.info(f"Random seed set to {seed} for reproducibility")


def setup_logging(log_file: Optional[str] = None, level=logging.INFO):
    """
    Configure logging with both console and file handlers.
    
    Args:
        log_file: Optional log file path
        level: Logging level
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_path = ARTIFACTS_DIR / log_file
        handlers.append(logging.FileHandler(log_path))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def save_metrics(metrics: Dict[str, Any], filename: str):
    """
    Save metrics dictionary to JSON file.
    
    Args:
        metrics: Dictionary of metrics
        filename: Output filename
    """
    output_path = ARTIFACTS_DIR / filename
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logging.info(f"Metrics saved to {output_path}")


def load_metrics(filename: str) -> Dict[str, Any]:
    """
    Load metrics from JSON file.
    
    Args:
        filename: Input filename
        
    Returns:
        Dictionary of metrics
    """
    input_path = ARTIFACTS_DIR / filename
    with open(input_path, 'r') as f:
        return json.load(f)


def save_plot(fig, filename: str, dpi: int = 300):
    """
    Save matplotlib figure to artifacts directory.
    
    Args:
        fig: Matplotlib figure object
        filename: Output filename
        dpi: Resolution in dots per inch
    """
    output_path = ARTIFACTS_DIR / filename
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Plot saved to {output_path}")


def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) for drift detection.
    
    PSI measures the shift between two distributions:
    - PSI < 0.1: No significant shift
    - 0.1 < PSI < 0.2: Moderate shift
    - PSI > 0.2: Significant shift (retrain recommended)
    
    Args:
        expected: Reference distribution (training data)
        actual: Current distribution (production data)
        bins: Number of bins for discretization
        
    Returns:
        PSI value
    """
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)  # Remove duplicates
    
    # Handle edge case where all values are the same
    if len(breakpoints) == 1:
        return 0.0
    
    # Calculate frequencies
    expected_freq = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_freq = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    expected_freq = expected_freq + epsilon
    actual_freq = actual_freq + epsilon
    
    # Calculate PSI
    psi = np.sum((actual_freq - expected_freq) * np.log(actual_freq / expected_freq))
    
    return psi


def set_style():
    """Set consistent plotting style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10


class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.time() - self.start_time
        logging.info(f"{self.name} completed in {self.elapsed:.3f}s")


# Import time for Timer class
import time

if __name__ == "__main__":
    # Test utilities
    set_seed()
    setup_logging("test.log")
    logging.info("Utilities module loaded successfully")
    
    # Test PSI calculation
    dist1 = np.random.normal(0, 1, 1000)
    dist2 = np.random.normal(0.2, 1.1, 1000)
    psi = calculate_psi(dist1, dist2)
    print(f"PSI between distributions: {psi:.4f}")
