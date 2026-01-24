#!/usr/bin/env python
# coding: utf-8

import os
import time
import random
import numpy as np
import torch
import functools
import gymnasium as gym
from typing import Callable, Any

from .preprocessing import PreprocessingWrapper


# Action mapping functions for Pong environment
def map_env_action(action: int) -> int:
    """
    Map environment action (6 actions) to reduced action space (3 actions).
    
    Pong has 6 discrete actions:
    - 0: NOOP, 1: FIRE -> mapped to 0 (NOOP)
    - 3: DOWN, 5: LEFTFIRE -> mapped to 3 (DOWN)
    - 2: UP, 4: RIGHTFIRE -> mapped to 2 (UP)
    
    Args:
        action: Environment action (0-5)
        
    Returns:
        Mapped action (0, 2, or 3)
    """
    match action:
        case 0 | 1:  # NOOP | FIRE
            return 0
        case 3 | 5:  # DOWN (LEFT | LEFTFIRE)
            return 3
        case 2 | 4:  # UP (RIGHT | RIGHTFIRE)
            return 2
        case _:
            raise ValueError(f"Invalid action: {action}")


def map_model_action(model_action: int) -> int:
    """
    Map model action back to environment action.
    
    Args:
        model_action: Model action (0, 1, or 2)
        
    Returns:
        Environment action (0, 2, or 3)
    """
    match model_action:
        case 0:
            return 0  # NOOP
        case 1:
            return 3  # DOWN (LEFT)
        case 2:
            return 2  # UP (RIGHT)
        case _:
            raise Exception(f"Invalid model_action: {model_action}")


def map_env2model(action: int) -> int:
    """
    Map environment action to model action index.
    
    Args:
        action: Environment action (0, 2, or 3)
        
    Returns:
        Model action index (0, 1, or 2)
    """
    match action:
        case 0:  # NOOP | FIRE
            return 0  # NOOP
        case 3:  # DOWN (LEFT | LEFTFIRE)
            return 1  # DOWN (LEFT)
        case 2:  # UP (RIGHT | RIGHTFIRE)
            return 2  # UP (RIGHT)
        case _:
            raise ValueError(f"Invalid action: {action}")


# Timing decorator
def timeit(func: Callable) -> Callable:
    """
    Decorator to measure function execution time.

    Args:
        func: Function to measure

    Returns:
        Wrapped function that prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        print(f"{func.__name__}() executed in {elapsed_time:.4f} seconds")
        return result

    return wrapper


# Random seed management
def set_config_seed(seed: int):
    """
    Set seeds for reproduction across random number generators.

    Ensures deterministic behavior across:
    - Python random module
    - NumPy
    - PyTorch (CPU and GPU)
    - CUDA backends

    Args:
        seed: Random seed to set
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


# Environment creation helper
def make_wrapped_env(env_name, **params):
    """
    Create a wrapped environment with preprocessing.

    Args:
        env_name: Name of the Gymnasium environment
        **params: Additional parameters for environment creation

    Returns:
        Preprocessed environment
    """
    render_mode = params.get("render_mode", "rgb_array")
    env = gym.make(env_name, 
                   render_mode=render_mode,
                   **params)
    env.reset()
    env = PreprocessingWrapper(env)
    return env


# Visualization helpers
def get_visual_array(array, show_last_batch=True):
    """
    Convert tensor or array to visual format for display.

    Args:
        array: Input tensor or numpy array
        show_last_batch: Whether to show the last batch item

    Returns:
        Tuple of (array, colormap) for visualization
    """
    if isinstance(array, torch.Tensor):
        if array.dim() == 4:  # [batch, channels, H, W]
            idx = -1 if show_last_batch else 0
            array = array[idx]
        if array.dim() == 3:  # [channels, H, W] 
            array = array.permute(1, 2, 0)  # [H, W, channels]
        array = array.detach().cpu().numpy()
        return array, "gray"
    elif isinstance(array, np.ndarray):
        if array.ndim == 4:  # [batch, channels, H, W]
            idx = -1 if show_last_batch else 0
            array = array[idx][:,:,-1]
        if array.ndim == 3:  # [channels, H, W] 
            array = array[:,:,-1]  # [H, W, channels]
        return array, "gray"

    return array, None


def debug_concat_frames(tensor: torch.Tensor):
    """
    Debug function to visualize concatenated frames.

    Useful for debugging preprocessing issues.

    Args:
        tensor: Input tensor of shape [4, 84, 84]
    """
    import matplotlib.pyplot as plt

    tensor = tensor.detach().cpu()
    fig, axes = plt.subplots(1, 4, figsize=(12, 12))

    for i, ax in enumerate(axes.flat):
        frame = tensor[i]  # shape (84, 84)
        ax.imshow(frame, cmap='gray')
        ax.set_title(f"Frame {i}")
        ax.axis('off')

    plt.tight_layout()
    plt.show()
