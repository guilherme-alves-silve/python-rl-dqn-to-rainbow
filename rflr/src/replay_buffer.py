#!/usr/bin/env python
# coding: utf-8

import numpy as np
import torch
import logging
from typing import NamedTuple, Tuple
from collections import deque


class Experience(NamedTuple):
    """
    A single experience tuple for reinforcement learning.
    
    Stores state, action, reward, next_state, and done flag.
    Uses optimized data types to minimize memory usage:
    - states: uint8 (0-255 for image pixels)
    - actions: uint8 (0-255 for discrete actions)
    - rewards: float32 (need precision for gradients)
    - done: bool
    """
    state: torch.ByteTensor  # uint8
    action: torch.ByteTensor  # uint8
    reward: torch.FloatTensor
    next_state: torch.ByteTensor  # uint8
    done: torch.BoolTensor


class ReplayBuffer:
    """
    Experience replay buffer for Deep Q-Learning.
    
    Stores experiences and samples them uniformly for training.
    Uses contiguous memory allocation for efficient batch sampling.
    """

    def __init__(self, 
                 size: int, 
                 state_shape: torch.Size, 
                 batch_size: int, 
                 device: torch.device,
                 map_env2model=lambda action: action,
                 _load_from_file=False):
        """
        Initialize the replay buffer.
        
        Args:
            size: Maximum number of experiences to store
            state_shape: Shape of the state tensor
            batch_size: Number of experiences to sample at once
            device: Device to move samples to (CPU/GPU)
            map_env2model: Function to map environment actions to model actions
            _load_from_file: Internal flag for loading from checkpoint
        """
        if _load_from_file:
            logging.info("Loading ReplayBuffer from file")
            return

        self.size = size
        self.batch_size = batch_size
        self.pos = 0
        self.count = 0
        self.device = device
        self.map_env2model = map_env2model
        
        # Pre-allocate tensors for efficient storage
        # states and next_states have shape [B, Concatenate, C, H, W]
        self.states = torch.zeros((size,) + state_shape, dtype=torch.uint8)
        self.actions = torch.zeros(size, dtype=torch.uint8)
        self.rewards = torch.zeros(size, dtype=torch.float32)
        self.next_states = torch.zeros((size,) + state_shape, dtype=torch.uint8)
        self.done = torch.zeros(size, dtype=torch.bool)

    def store(self, experience: Experience) -> None:
        """
        Store a single experience in the buffer.
        
        Uses circular buffer behavior - overwrites oldest experiences when full.
        
        Args:
            experience: Experience tuple (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = experience
        pos = self.pos
        size = self.size
        
        # Store experience at current position
        self.states[pos] = state
        self.actions[pos] = action
        self.rewards[pos] = reward
        self.next_states[pos] = next_state
        self.done[pos] = done
        
        # Update position circularly
        self.pos = (pos + 1) % size
        if self.count < size:
            self.count += 1

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of experiences for training.
        
        Converts stored uint8 data to float32 and moves to the specified device.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, done) tensors
        """
        # Sample random indices without replacement
        samples_pos = np.random.choice(self.count, self.batch_size, replace=False)
        
        # Convert to float32 and move to device
        states = self.states[samples_pos].to(self.device, dtype=torch.float32).div_(255.0)
        actions = torch.from_numpy(self._mapped_actions(samples_pos)).to(self.device, dtype=torch.long)
        rewards = self.rewards[samples_pos].to(self.device)
        next_states = self.next_states[samples_pos].to(self.device, dtype=torch.float32).div_(255.0)
        done = self.done[samples_pos].to(self.device)
        
        return states, actions, rewards, next_states, done

    def _mapped_actions(self, samples_pos):
        """
        Map environment actions back to model actions for sampling.
        
        Args:
            samples_pos: Array of indices to sample
            
        Returns:
            Mapped actions as numpy array
        """
        # We have to map back to model because we reduced the environment from 6 to 3 actions
        actions_np = self.actions[samples_pos].numpy()
        return np.array([self.map_env2model(a) for a in actions_np])

    def enough(self) -> bool:
        """
        Check if buffer has enough experiences for a batch.
        
        Returns:
            True if buffer contains at least batch_size experiences
        """
        return len(self) >= self.batch_size

    @property
    def cnt_rewards(self) -> int:
        """
        Count the number of non-zero rewards in the buffer.
        
        Returns:
            Number of experiences with positive reward
        """
        return torch.count_nonzero(self.rewards > 0).item()

    def __len__(self):
        """
        Get the current number of experiences in the buffer.
        
        Returns:
            Current count of experiences
        """
        return self.count

    # Checkpoint methods
    def save(self, file_path: str):
        """
        Save buffer to disk using PyTorch's native format.
        
        Args:
            file_path: Path to save the checkpoint
        """
        torch.save({
            'states': self.states,
            'actions': self.actions,
            'rewards': self.rewards,
            'next_states': self.next_states,
            'done': self.done,
            'pos': self.pos,
            'count': self.count,
            'size': self.size,
            'batch_size': self.batch_size
        }, file_path)

    def _set_checkpoint_data(self, checkpoint: dict):
        """
        Load checkpoint data into the buffer.
        
        Args:
            checkpoint: Dictionary containing checkpoint data
        """
        self.states = checkpoint['states']
        self.actions = checkpoint['actions']
        self.rewards = checkpoint['rewards']
        self.next_states = checkpoint['next_states']
        self.done = checkpoint['done']
        self.pos = checkpoint['pos']
        self.count = checkpoint['count']
        self.size = checkpoint['size']
        self.batch_size = checkpoint['batch_size']
        return self

    @classmethod
    def from_file(cls, 
                  file_path: str, 
                  device: torch.device,
                  map_env2model=lambda action: action):
        """
        Create buffer from saved file.
        
        Args:
            file_path: Path to the checkpoint file
            device: Device to use for the buffer
            map_env2model: Function to map environment actions to model actions
            
        Returns:
            Loaded ReplayBuffer instance
        """
        checkpoint = torch.load(file_path, map_location='cpu')
        buffer = cls(
            size=None,
            state_shape=None,
            batch_size=None,
            device=None,
            map_env2model=None,
            _load_from_file=True
        )
        buffer.device = device
        buffer.map_env2model = map_env2model
        buffer._set_checkpoint_data(checkpoint)
        return buffer


class BinaryReplayBuffer:
    """
    Binary replay buffer that separates wins and losses.
    
    Maintains a 50/50 balance between positive and non-positive reward experiences
    to ensure the agent learns from both success and failure.
    """
    
    def __init__(self,
                 size: int,
                 state_shape: torch.Size,
                 batch_size: int,
                 device: torch.device,
                 map_env2model=lambda action: action,
                 _load_from_file=False):
        """
        Initialize the binary replay buffer.
        
        Args:
            size: Maximum number of experiences to store
            state_shape: Shape of the state tensor
            batch_size: Number of experiences to sample at once
            device: Device to move samples to (CPU/GPU)
            map_env2model: Function to map environment actions to model actions
            _load_from_file: Internal flag for loading from checkpoint
        """
        if _load_from_file:
            logging.info("Loading BinaryReplayBuffer from file")
            return

        self.size = size
        self.batch_size = batch_size
        self.half_batch_size = batch_size // 2
        self.device = device
        self.enough_both = False

        # Split size evenly between win and loss buffers
        half_size = size // 2
        
        # Create separate buffers for wins and other experiences
        self.win_replay_buffer = ReplayBuffer(
            half_size, state_shape, self.half_batch_size, device,
            map_env2model, _load_from_file=_load_from_file
        )
        self.other_replay_buffer = ReplayBuffer(
            half_size, state_shape, self.half_batch_size, device,
            map_env2model, _load_from_file=_load_from_file
        )

    def store(self, experience: Experience, delta: float = 0.0001) -> None:
        """
        Store experience in the appropriate buffer based on reward.
        
        Args:
            experience: Experience tuple to store
            delta: Small value to determine positive reward threshold
        """
        reward = experience.reward
        if reward > delta:
            self.win_replay_buffer.store(experience)
        else:
            self.other_replay_buffer.store(experience)

    def sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a balanced batch from both buffers.
        
        Returns:
            Concatenated tuple of (states, actions, rewards, next_states, done)
        """
        # Sample from both buffers
        w_states, w_actions, w_rewards, w_next_states, w_done = self.win_replay_buffer.sample_batch()
        o_states, o_actions, o_rewards, o_next_states, o_done = self.other_replay_buffer.sample_batch()
        
        # Concatenate the samples
        return (
            torch.cat([w_states, o_states]),
            torch.cat([w_actions, o_actions]),
            torch.cat([w_rewards, o_rewards]),
            torch.cat([w_next_states, o_next_states]),
            torch.cat([w_done, o_done])
        )

    def enough(self) -> bool:
        """
        Check if both buffers have enough experiences.
        
        Returns:
            True if both buffers contain at least half_batch_size experiences
        """
        if self.enough_both:
            return True
        self.enough_both = (
            len(self.win_replay_buffer) >= self.half_batch_size and
            len(self.other_replay_buffer) >= self.half_batch_size
        )
        return self.enough_both

    def __len__(self):
        """
        Get total number of experiences across both buffers.
        
        Returns:
            Total count of experiences
        """
        return len(self.win_replay_buffer) + len(self.other_replay_buffer)

    @property
    def cnt_rewards(self) -> int:
        """
        Count total non-zero rewards across both buffers.
        
        Returns:
            Total count of positive rewards
        """
        return self.win_replay_buffer.cnt_rewards + self.other_replay_buffer.cnt_rewards

    def save(self, file_path: str) -> None:
        """
        Save both buffers to disk.
        
        Args:
            file_path: Base path for saving (will be modified for each buffer)
        """
        self.win_replay_buffer.save(file_path.replace(".pth", "_win.pth"))
        self.other_replay_buffer.save(file_path.replace(".pth", "_other.pth"))

    @classmethod
    def from_file(cls,
                  file_path: str,
                  device: torch.device,
                  map_env2model=lambda action: action):
        """
        Create buffer from saved file.
        
        Args:
            file_path: Base path to the checkpoint files
            device: Device to use for the buffer
            map_env2model: Function to map environment actions to model actions
            
        Returns:
            Loaded BinaryReplayBuffer instance
        """
        win_replay_buffer_path = file_path.replace(".pth", "_win.pth")
        other_replay_buffer_path = file_path.replace(".pth", "_other.pth")

        bin_replay_buffer = cls(
            size=None,
            state_shape=None,
            batch_size=None,
            device=None,
            map_env2model=map_env2model,
            _load_from_file=True
        )

        # Load both sub-buffers
        bin_replay_buffer.win_replay_buffer = ReplayBuffer.from_file(
            win_replay_buffer_path, device, map_env2model
        )
        bin_replay_buffer.other_replay_buffer = ReplayBuffer.from_file(
            other_replay_buffer_path, device, map_env2model
        )

        # Extract values from the loaded buffers
        bin_replay_buffer.size = (bin_replay_buffer.win_replay_buffer.size +
                                   bin_replay_buffer.other_replay_buffer.size)
        bin_replay_buffer.batch_size = (bin_replay_buffer.win_replay_buffer.batch_size +
                                         bin_replay_buffer.other_replay_buffer.batch_size)
        bin_replay_buffer.half_batch_size = bin_replay_buffer.batch_size // 2
        bin_replay_buffer.map_env2model = map_env2model
        bin_replay_buffer.device = device
        bin_replay_buffer.enough_both = False

        return bin_replay_buffer
