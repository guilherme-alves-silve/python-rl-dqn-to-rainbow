#!/usr/bin/env python
# coding: utf-8

import unittest
import torch
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing import PreprocessingWrapper
from unittest.mock import Mock, MagicMock


class TestPreprocessingWrapper(unittest.TestCase):
    """Unit tests for the PreprocessingWrapper class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock environment
        self.mock_env = Mock()
        self.mock_env.observation_space = Mock()
        self.mock_env.action_space = Mock()
        
        # Create wrapper instance
        self.wrapper = PreprocessingWrapper(
            env=self.mock_env,
            skip=4,
            resize=84,
            concatenate=4,
            interpolation_mode="nearest"
        )
        
    def test_wrapper_initialization(self):
        """Test that wrapper is properly initialized."""
        self.assertEqual(self.wrapper._skip, 4)
        self.assertEqual(self.wrapper._resize, 84)
        self.assertEqual(self.wrapper._concatenate, 4)
        self.assertEqual(self.wrapper._interpolation_mode, "nearest")
        self.assertEqual(self.wrapper.env, self.mock_env)
        
    def test_grayscale_frame(self):
        """Test grayscale conversion."""
        # Create RGB frame [210, 160, 3]
        rgb_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
        
        # Convert to grayscale
        gray_tensor = self.wrapper._grayscale_frame(rgb_frame)
        
        # Check output shape [1, 210, 160]
        self.assertEqual(gray_tensor.shape, (1, 210, 160))
        
        # Check dtype
        self.assertEqual(gray_tensor.dtype, torch.uint8)
        
        # Check that values are reasonable (should be between 0 and 255)
        self.assertTrue(torch.all(gray_tensor >= 0))
        self.assertTrue(torch.all(gray_tensor <= 255))
        
    def test_resize_frame(self):
        """Test frame resizing."""
        # Create grayscale tensor [1, 210, 160]
        gray_tensor = torch.randint(0, 255, (1, 210, 160), dtype=torch.uint8)
        
        # Resize to 84x84
        resized_tensor = self.wrapper._resize_frame(gray_tensor)
        
        # Check output shape [1, 84, 84]
        self.assertEqual(resized_tensor.shape, (1, 84, 84))
        
        # Check dtype (should remain uint8)
        self.assertEqual(resized_tensor.dtype, torch.uint8)
        
    def test_skip_frames(self):
        """Test frame skipping."""
        # Mock environment step to return different states and rewards
        step_count = 0
        def mock_step(action):
            nonlocal step_count
            step_count += 1
            # Return different rewards and states
            state = np.full((210, 160, 3), step_count, dtype=np.uint8)
            reward = float(step_count)
            term = step_count >= 3
            trunc = False
            info = {}
            return state, reward, term, trunc, info
            
        self.mock_env.step = mock_step
        
        # Skip frames
        state, total_reward, term, trunc, info = self.wrapper._skip_frames(0)
        
        # Check that step was called multiple times
        self.assertGreaterEqual(step_count, 4)  # skip + 1
        
        # Check accumulated reward
        self.assertGreater(total_reward, 0)
        
        # Check state shape
        self.assertEqual(state.shape, (210, 160, 3))
        
    def test_finish_concatenate_frames(self):
        """Test frame concatenation."""
        # Create list of frames
        frames = []
        for i in range(2):  # Only 2 frames
            frame = torch.full((1, 84, 84), i, dtype=torch.uint8)
            frames.append(frame)
            
        rewards = [1.0, 2.0]
        
        # Concatenate frames (should pad to 4 frames)
        concat_state, total_reward = self.wrapper._finish_concatenate_frames(frames, rewards)
        
        # Check output shape [4, 84, 84]
        self.assertEqual(concat_state.shape, (4, 84, 84))
        
        # Check that frames were padded
        torch.testing.assert_close(concat_state[0], torch.full((84, 84), 0, dtype=torch.uint8))
        torch.testing.assert_close(concat_state[1], torch.full((84, 84), 1, dtype=torch.uint8))
        # Last two should be copies of the last frame
        torch.testing.assert_close(concat_state[2], torch.full((84, 84), 1, dtype=torch.uint8))
        torch.testing.assert_close(concat_state[3], torch.full((84, 84), 1, dtype=torch.uint8))
        
        # Check total reward
        self.assertEqual(total_reward, 3.0)
        
    def test_concatenate_exact_frames(self):
        """Test concatenation when we have exactly the right number of frames."""
        # Create exactly 4 frames
        frames = []
        for i in range(4):
            frame = torch.full((1, 84, 84), i, dtype=torch.uint8)
            frames.append(frame)
            
        rewards = [1.0, 2.0, 3.0, 4.0]
        
        # Concatenate
        concat_state, total_reward = self.wrapper._finish_concatenate_frames(frames, rewards)
        
        # Check output shape
        self.assertEqual(concat_state.shape, (4, 84, 84))
        
        # Check each frame
        for i in range(4):
            torch.testing.assert_close(concat_state[i], torch.full((84, 84), i, dtype=torch.uint8))
            
        # Check total reward
        self.assertEqual(total_reward, 10.0)
        
    def test_reset(self):
        """Test environment reset."""
        # Mock reset to return initial state
        def mock_reset(seed=None, options=None):
            state = np.ones((210, 160, 3), dtype=np.uint8)
            info = {}
            return state, info
            
        self.mock_env.reset = mock_reset
        
        # Reset wrapper
        state, info = self.wrapper.reset()
        
        # Check output shape
        self.assertEqual(state.shape, (4, 84, 84))
        
        # Check dtype
        self.assertEqual(state.dtype, torch.uint8)
        
    def test_step_integration(self):
        """Test complete step integration."""
        # Setup mock environment
        step_count = 0
        def mock_step(action):
            nonlocal step_count
            step_count += 1
            state = np.full((210, 160, 3), step_count, dtype=np.uint8)
            reward = 1.0
            term = step_count >= 10
            trunc = False
            info = {}
            return state, reward, term, trunc, info
            
        self.mock_env.step = mock_step
        
        # Step wrapper
        state, reward, term, trunc, info = self.wrapper.step(0)
        
        # Check outputs
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertEqual(state.dtype, torch.uint8)
        self.assertGreater(reward, 0)
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        
    def test_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        # Create RGB frames with known values
        def mock_step(action):
            # Create frame with specific pattern
            state = np.zeros((210, 160, 3), dtype=np.uint8)
            state[:, :, 0] = 255  # Red channel max
            state[:, :, 1] = 128  # Green channel half
            state[:, :, 2] = 64   # Blue channel quarter
            reward = 1.0
            term = False
            trunc = False
            info = {}
            return state, reward, term, trunc, info
            
        self.mock_env.step = mock_step
        
        # Process through pipeline
        state, reward, term, trunc, info = self.wrapper.step(0)
        
        # Check final output
        self.assertEqual(state.shape, (4, 84, 84))
        self.assertEqual(state.dtype, torch.uint8)
        
        # All frames should be the same (since we used same input)
        for i in range(1, 4):
            torch.testing.assert_close(state[0], state[i])
            
    def test_grayscale_calculation(self):
        """Test that grayscale conversion works correctly."""
        # Create RGB frame with known values
        rgb_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        rgb_frame[:, :, 0] = 255  # Red
        rgb_frame[:, :, 1] = 0    # Green  
        rgb_frame[:, :, 2] = 0    # Blue
        
        # Expected grayscale: (255 + 0 + 0) / 3 = 85
        gray_tensor = self.wrapper._grayscale_frame(rgb_frame)
        expected_value = 255.0 / 3.0  # Mean of RGB channels
        
        # Check that values are close to expected
        mean_val = gray_tensor.float().mean().item()
        self.assertAlmostEqual(mean_val, expected_value, places=1)
        
    def test_padding_behavior(self):
        """Test padding when episode ends early."""
        # Create only 2 frames (less than concatenate=4)
        frames = []
        frames.append(torch.full((1, 84, 84), 10, dtype=torch.uint8))
        frames.append(torch.full((1, 84, 84), 20, dtype=torch.uint8))
        
        rewards = [5.0, 10.0]
        
        # Concatenate should pad with last frame
        concat_state, total_reward = self.wrapper._finish_concatenate_frames(frames, rewards)
        
        # Check all frames
        torch.testing.assert_close(concat_state[0], torch.full((84, 84), 10, dtype=torch.uint8))
        torch.testing.assert_close(concat_state[1], torch.full((84, 84), 20, dtype=torch.uint8))
        torch.testing.assert_close(concat_state[2], torch.full((84, 84), 20, dtype=torch.uint8))
        torch.testing.assert_close(concat_state[3], torch.full((84, 84), 20, dtype=torch.uint8))
        
        self.assertEqual(total_reward, 15.0)


if __name__ == '__main__':
    unittest.main()
