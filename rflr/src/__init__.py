#!/usr/bin/env python
# coding: utf-8

"""
Deep Q-Network Implementation for Atari Pong

This package contains a refactored implementation of Deep Q-Network
for playing Atari Pong, based on the Nature paper:
"Human-level Control Through Deep Reinforcement Learning"

Modules:
    agent: Main DQN agent class
    network: Deep Q-Network architecture
    replay_buffer: Experience replay buffer implementation
    preprocessing: Environment preprocessing wrapper
    utils: Utility functions and helpers
"""

from .agent import AgentDQN
from .network import Network
from .replay_buffer import ReplayBuffer, BinaryReplayBuffer, Experience
from .preprocessing import PreprocessingWrapper
from .utils import (
    map_env_action, 
    map_model_action, 
    map_env2model,
    timeit,
    set_config_seed,
    make_wrapped_env,
    get_visual_array,
    debug_concat_frames
)

__version__ = "1.0.0"
__author__ = "Refactored from original by Guilherme Alves Silveira"

__all__ = [
    "AgentDQN",
    "Network", 
    "ReplayBuffer",
    "BinaryReplayBuffer",
    "Experience",
    "PreprocessingWrapper",
    "map_env_action",
    "map_model_action", 
    "map_env2model",
    "timeit",
    "set_config_seed",
    "make_wrapped_env",
    "get_visual_array",
    "debug_concat_frames"
]
