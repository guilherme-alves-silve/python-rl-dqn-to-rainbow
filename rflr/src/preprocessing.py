#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn.functional as F
import gymnasium as gym
from gymnasium import Env
from typing import Tuple


class PreprocessingWrapper(gym.Wrapper):
    """
    Environment wrapper that preprocesses Atari frames for Deep Q-Learning.

    Performs the following preprocessing steps:
    1. Frame skipping - skips similar frames to help the network notice changes
    2. Grayscale conversion - reduces from 3 channels to 1 for faster processing
    3. Resizing - reduces frame size from 210x160 to 84x84
    4. Frame stacking - concatenates multiple frames to capture motion
    """

    def __init__(self,
                 env: Env,
                 skip=4,
                 resize=84,
                 concatenate=4,
                 interpolation_mode="nearest"):
        """
        Initialize the preprocessing wrapper.

        Args:
            env: The Gymnasium environment to wrap
            skip: Number of frames to skip between processed frames
            resize: Target size for resizing (both height and width)
            concatenate: Number of frames to concatenate
            interpolation_mode: Interpolation method for resizing
        """
        super().__init__(env)
        self._skip = skip
        self._resize = resize
        self._concatenate = concatenate
        self._interpolation_mode = interpolation_mode

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, dict]:
        """
        Execute action in environment and return preprocessed result.

        Args:
            action: Action to take in the environment

        Returns:
            Tuple of (processed_state, total_reward, terminated, truncated, info)
        """
        frames = []
        rewards = []

        # Collect multiple frames (with skipping) for concatenation
        for _ in range(self._concatenate):
            state, reward, term, trunc, info = self._skip_frames(action)
            gray_state = self._grayscale_frame(state)
            resized_state = self._resize_frame(gray_state)
            frames.append(resized_state)
            rewards.append(reward)

            if term or trunc:
                break

        # Concatenate frames and sum rewards
        concat_state, total_reward = self._finish_concatenate_frames(frames, rewards)
        return concat_state, total_reward, term, trunc, info

    def _skip_frames(self, action: int):
        """
        Skip frames to get meaningful changes between images.
        
        This helps the network notice differences between consecutive states.
        Without skipping, consecutive frames would be too similar, making
        it difficult for the network to learn meaningful patterns.
        
        Args:
            action: Action to repeat during skipped frames
            
        Returns:
            Tuple of (state, total_reward, terminated, truncated, info)
        """
        total_reward = 0
        for _ in range(self._skip + 1):
            state, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if term or trunc:
                break

        return state, total_reward, term, trunc, info

    def _grayscale_frame(self, state):
        """
        Convert RGB frame to grayscale.
        
        Reduces processing from 3 channels to 1 channel.
        - Original: 3 x 210 x 160 = 100,800 values
        - Grayscale: 1 x 210 x 160 = 33,600 values
        
        Args:
            state: RGB state array from environment
            
        Returns:
            Grayscale state tensor
        """
        # Convert to PyTorch tensor
        state = torch.from_numpy(state).float()
        
        # Grayscale conversion: take mean across color channels
        # Input: [Height, Width, Channels]
        # Output: [Channels, Height, Width] with single channel
        state = (state.mean(dim=2, keepdim=True)
                      .permute(2, 0, 1)
                      .to(dtype=torch.uint8))
        return state

    def _resize_frame(self, state):
        """
        Resize frame to smaller dimensions for faster processing.
        
        Reduces from original size to 84x84 (79% reduction):
        - From: 210 x 160 = 33,600 values
        - To: 84 x 84 = 7,056 values
        
        Args:
            state: Grayscale state tensor
            
        Returns:
            Resized state tensor
        """
        # Add batch dimension, resize, then remove batch dimension
        return (F.interpolate(state.unsqueeze(dim=0),
                             size=(self._resize, self._resize),
                             mode=self._interpolation_mode)).squeeze(dim=0)

    def _finish_concatenate_frames(self, frames, rewards):
        """
        Concatenate multiple frames to capture motion information.
        
        The neural network needs to see motion to learn how to react over time.
        Without concatenation, it would be difficult to learn ball trajectory
        and paddle movement patterns.
        
        Ensures the final shape is concatenate x 84 x 84.
        If we don't have enough frames (e.g., episode ended early),
        we pad with the last available frame.
        
        Args:
            frames: List of processed frames
            rewards: List of rewards for each frame
            
        Returns:
            Tuple of (concatenated_state, total_reward)
        """
        # Pad with last frame if we don't have enough
        for i in range(len(frames), self._concatenate):
            last_state = frames[-1]
            last_reward = rewards[-1]
            # Clone to maintain consistency
            frames.append(last_state.clone())
            rewards.append(last_reward)
        
        # Concatenate frames along channel dimension
        # Result: [concatenate, height, width]
        state = torch.cat(frames, dim=0)
        total_reward = sum(rewards)
        return state, total_reward

    def reset(self, *, seed=None, options=None) -> Tuple[torch.Tensor, dict]:
        """
        Reset the environment and return initial preprocessed state.
        
        Args:
            seed: Random seed for environment
            options: Options for environment reset
            
        Returns:
            Tuple of (initial_processed_state, info)
        """
        state, info = self.env.reset(seed=seed, options=options)

        frames = []
        rewards = []

        # Process initial frame
        gray_state = self._grayscale_frame(state)
        resized_state = self._resize_frame(gray_state)

        frames.append(resized_state)
        rewards.append(0.0)

        # Concatenate to get proper initial state shape
        concat_state, _ = self._finish_concatenate_frames(frames, rewards)
        return concat_state, info
