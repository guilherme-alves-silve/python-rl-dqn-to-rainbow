#!/usr/bin/env python
# coding: utf-8

import unittest
import torch
import numpy as np
import tempfile
import os
import sys

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from replay_buffer import ReplayBuffer, BinaryReplayBuffer, Experience


class TestReplayBuffer(unittest.TestCase):
    """Unit tests for the ReplayBuffer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.buffer_size = 100
        self.batch_size = 32
        self.state_shape = torch.Size([4, 84, 84])  # 4 frames, 84x84 each
        
        # Create buffer instance
        self.buffer = ReplayBuffer(
            size=self.buffer_size,
            state_shape=self.state_shape,
            batch_size=self.batch_size,
            device=self.device,
            map_env2model=lambda x: x  # Identity function for testing
        )
        
    def test_buffer_initialization(self):
        """Test that buffer is properly initialized."""
        self.assertEqual(self.buffer.size, self.buffer_size)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
        self.assertEqual(self.buffer.pos, 0)
        self.assertEqual(self.buffer.count, 0)
        self.assertEqual(self.buffer.device, self.device)
        
        # Check pre-allocated tensors
        self.assertEqual(self.buffer.states.shape, (self.buffer_size,) + self.state_shape)
        self.assertEqual(self.buffer.actions.shape, (self.buffer_size,))
        self.assertEqual(self.buffer.rewards.shape, (self.buffer_size,))
        self.assertEqual(self.buffer.next_states.shape, (self.buffer_size,) + self.state_shape)
        self.assertEqual(self.buffer.done.shape, (self.buffer_size,))
        
        # Check data types
        self.assertEqual(self.buffer.states.dtype, torch.uint8)
        self.assertEqual(self.buffer.actions.dtype, torch.uint8)
        self.assertEqual(self.buffer.rewards.dtype, torch.float32)
        self.assertEqual(self.buffer.next_states.dtype, torch.uint8)
        self.assertEqual(self.buffer.done.dtype, torch.bool)
        
    def test_store_single_experience(self):
        """Test storing a single experience."""
        # Create test experience
        state = torch.randint(0, 255, self.state_shape, dtype=torch.uint8)
        action = torch.tensor(1, dtype=torch.uint8)
        reward = torch.tensor(1.0, dtype=torch.float32)
        next_state = torch.randint(0, 255, self.state_shape, dtype=torch.uint8)
        done = torch.tensor(False, dtype=torch.bool)
        
        experience = Experience(state, action, reward, next_state, done)
        
        # Store experience
        self.buffer.store(experience)
        
        # Check buffer state
        self.assertEqual(self.buffer.count, 1)
        self.assertEqual(self.buffer.pos, 1)
        
        # Check stored data
        torch.testing.assert_close(self.buffer.states[0], state)
        torch.testing.assert_close(self.buffer.actions[0], action)
        torch.testing.assert_close(self.buffer.rewards[0], reward)
        torch.testing.assert_close(self.buffer.next_states[0], next_state)
        torch.testing.assert_close(self.buffer.done[0], done)
        
    def test_store_multiple_experiences(self):
        """Test storing multiple experiences."""
        num_experiences = 10
        
        for i in range(num_experiences):
            state = torch.full(self.state_shape, i, dtype=torch.uint8)
            action = torch.tensor(i % 3, dtype=torch.uint8)
            reward = torch.tensor(float(i), dtype=torch.float32)
            next_state = torch.full(self.state_shape, i + 1, dtype=torch.uint8)
            done = torch.tensor(i % 2 == 0, dtype=torch.bool)
            
            experience = Experience(state, action, reward, next_state, done)
            self.buffer.store(experience)
        
        # Check buffer state
        self.assertEqual(self.buffer.count, num_experiences)
        self.assertEqual(self.buffer.pos, num_experiences)
        
        # Check some stored values
        torch.testing.assert_close(self.buffer.states[5], torch.full(self.state_shape, 5, dtype=torch.uint8))
        torch.testing.assert_close(self.buffer.rewards[5], torch.tensor(5.0))
        
    def test_circular_buffer_behavior(self):
        """Test that buffer overwrites old experiences when full."""
        # Fill buffer completely
        for i in range(self.buffer_size):
            state = torch.full(self.state_shape, i, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(experience)
        
        # Buffer should be full
        self.assertEqual(self.buffer.count, self.buffer_size)
        self.assertEqual(self.buffer.pos, 0)  # Wrapped around
        
        # Store one more experience (should overwrite first)
        new_state = torch.full(self.state_shape, 999, dtype=torch.uint8)
        new_experience = Experience(new_state, torch.tensor(1), torch.tensor(1.0), new_state, torch.tensor(True))
        self.buffer.store(new_experience)
        
        # Check that first experience was overwritten
        self.assertEqual(self.buffer.count, self.buffer_size)
        self.assertEqual(self.buffer.pos, 1)
        torch.testing.assert_close(self.buffer.states[0], new_state)
        
    def test_enough_method(self):
        """Test the enough() method."""
        # Initially not enough
        self.assertFalse(self.buffer.enough())
        
        # Fill buffer to batch size
        for i in range(self.batch_size):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(experience)
        
        # Now should have enough
        self.assertTrue(self.buffer.enough())
        
    def test_len_method(self):
        """Test the __len__ method."""
        self.assertEqual(len(self.buffer), 0)
        
        # Add some experiences
        for i in range(5):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(experience)
            
        self.assertEqual(len(self.buffer), 5)
        
    def test_cnt_rewards_property(self):
        """Test the cnt_rewards property."""
        # Add experiences with some positive rewards
        rewards = [0.0, 1.0, -1.0, 2.0, 0.0, 0.5, 0.0]
        
        for reward in rewards:
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(reward), state, torch.tensor(False))
            self.buffer.store(experience)
        
        # Should count 3 positive rewards (1.0, 2.0, 0.5)
        self.assertEqual(self.buffer.cnt_rewards, 3)
        
    def test_sample_batch_shape(self):
        """Test that sampled batch has correct shape."""
        # Fill buffer with enough experiences
        for i in range(self.batch_size + 10):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(experience)
        
        # Sample batch
        states, actions, rewards, next_states, done = self.buffer.sample_batch()
        
        # Check shapes
        self.assertEqual(states.shape, (self.batch_size,) + self.state_shape)
        self.assertEqual(actions.shape, (self.batch_size,))
        self.assertEqual(rewards.shape, (self.batch_size,))
        self.assertEqual(next_states.shape, (self.batch_size,) + self.state_shape)
        self.assertEqual(done.shape, (self.batch_size,))
        
    def test_sample_batch_device(self):
        """Test that sampled batch is on correct device."""
        # Fill buffer
        for i in range(self.batch_size + 10):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(experience)
        
        # Sample batch
        states, actions, rewards, next_states, done = self.buffer.sample_batch()
        
        # Check devices
        self.assertEqual(states.device, self.device)
        self.assertEqual(actions.device, self.device)
        self.assertEqual(rewards.device, self.device)
        self.assertEqual(next_states.device, self.device)
        self.assertEqual(done.device, self.device)
        
    def test_sample_batch_normalization(self):
        """Test that states are normalized to [0, 1] range."""
        # Add experience with maximum uint8 values
        state = torch.full(self.state_shape, 255, dtype=torch.uint8)
        experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
        self.buffer.store(experience)
        
        # Add more experiences to ensure we can sample
        for i in range(self.batch_size):
            exp = Experience(torch.zeros(self.state_shape, dtype=torch.uint8), 
                            torch.tensor(0), torch.tensor(0.0), 
                            torch.zeros(self.state_shape, dtype=torch.uint8), 
                            torch.tensor(False))
            self.buffer.store(exp)
        
        # Sample batch
        states, _, _, _, _ = self.buffer.sample_batch()
        
        # Check normalization
        self.assertTrue(torch.all(states >= 0.0))
        self.assertTrue(torch.all(states <= 1.0))
        
        # The 255 values should become 1.0
        max_val = states.max().item()
        self.assertAlmostEqual(max_val, 1.0, places=5)
        
    def test_action_mapping(self):
        """Test that actions are properly mapped."""
        # Create buffer with custom mapping function
        def custom_map(action):
            return action + 1  # Simple mapping for testing
            
        buffer = ReplayBuffer(
            size=100,
            state_shape=self.state_shape,
            batch_size=32,
            device=self.device,
            map_env2model=custom_map
        )
        
        # Add experience with action 0
        state = torch.zeros(self.state_shape, dtype=torch.uint8)
        experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
        buffer.store(experience)
        
        # Add more experiences
        for i in range(buffer.batch_size):
            exp = Experience(state, torch.tensor(1), torch.tensor(0.0), state, torch.tensor(False))
            buffer.store(exp)
        
        # Sample batch - action 0 should be mapped to 1
        _, actions, _, _, _ = buffer.sample_batch()
        
        # Check that at least one action was mapped correctly
        self.assertTrue(torch.any(actions == 1))
        
    def test_save_and_load(self):
        """Test saving and loading buffer from file."""
        # Add some experiences
        for i in range(10):
            state = torch.full(self.state_shape, i, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(i), torch.tensor(float(i)), state, torch.tensor(i % 2 == 0))
            self.buffer.store(experience)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            tmp_path = tmp.name
            
        try:
            self.buffer.save(tmp_path)
            
            # Load from file
            loaded_buffer = ReplayBuffer.from_file(
                tmp_path, 
                device=self.device,
                map_env2model=lambda x: x
            )
            
            # Check loaded data
            self.assertEqual(len(loaded_buffer), 10)
            torch.testing.assert_close(loaded_buffer.states[5], torch.full(self.state_shape, 5, dtype=torch.uint8))
            self.assertEqual(loaded_buffer.cnt_rewards, 5)  # 0,1,2,3,4 are positive
            
        finally:
            # Clean up
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestBinaryReplayBuffer(unittest.TestCase):
    """Unit tests for the BinaryReplayBuffer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.buffer_size = 100
        self.batch_size = 32
        self.state_shape = torch.Size([4, 84, 84])
        
        # Create buffer instance
        self.buffer = BinaryReplayBuffer(
            size=self.buffer_size,
            state_shape=self.state_shape,
            batch_size=self.batch_size,
            device=self.device,
            map_env2model=lambda x: x
        )
        
    def test_binary_buffer_initialization(self):
        """Test that binary buffer is properly initialized."""
        self.assertEqual(self.buffer.size, self.buffer_size)
        self.assertEqual(self.buffer.batch_size, self.batch_size)
        self.assertEqual(self.buffer.half_batch_size, self.batch_size // 2)
        self.assertFalse(self.buffer.enough_both)
        
        # Check sub-buffers
        self.assertIsInstance(self.buffer.win_replay_buffer, ReplayBuffer)
        self.assertIsInstance(self.buffer.other_replay_buffer, ReplayBuffer)
        
        # Check sizes are split
        self.assertEqual(self.buffer.win_replay_buffer.size, self.buffer_size // 2)
        self.assertEqual(self.buffer.other_replay_buffer.size, self.buffer_size // 2)
        
    def test_store_positive_reward(self):
        """Test storing experience with positive reward."""
        state = torch.zeros(self.state_shape, dtype=torch.uint8)
        experience = Experience(state, torch.tensor(0), torch.tensor(1.0), state, torch.tensor(False))
        
        self.buffer.store(experience)
        
        # Should be in win buffer
        self.assertEqual(len(self.buffer.win_replay_buffer), 1)
        self.assertEqual(len(self.buffer.other_replay_buffer), 0)
        
    def test_store_zero_reward(self):
        """Test storing experience with zero reward."""
        state = torch.zeros(self.state_shape, dtype=torch.uint8)
        experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
        
        self.buffer.store(experience)
        
        # Should be in other buffer
        self.assertEqual(len(self.buffer.win_replay_buffer), 0)
        self.assertEqual(len(self.buffer.other_replay_buffer), 1)
        
    def test_store_negative_reward(self):
        """Test storing experience with negative reward."""
        state = torch.zeros(self.state_shape, dtype=torch.uint8)
        experience = Experience(state, torch.tensor(0), torch.tensor(-1.0), state, torch.tensor(False))
        
        self.buffer.store(experience)
        
        # Should be in other buffer
        self.assertEqual(len(self.buffer.win_replay_buffer), 0)
        self.assertEqual(len(self.buffer.other_replay_buffer), 1)
        
    def test_enough_method_both_buffers(self):
        """Test enough() method requires both buffers to have samples."""
        # Initially not enough
        self.assertFalse(self.buffer.enough())
        self.assertFalse(self.buffer.enough_both)
        
        # Fill only win buffer
        for i in range(self.batch_size):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(1.0), state, torch.tensor(False))
            self.buffer.store(experience)
            
        # Still not enough
        self.assertFalse(self.buffer.enough())
        
        # Fill other buffer
        for i in range(self.batch_size):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            experience = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(experience)
            
        # Now should have enough
        self.assertTrue(self.buffer.enough())
        self.assertTrue(self.buffer.enough_both)
        
    def test_sample_batch_balanced(self):
        """Test that sample batch is balanced between win and other."""
        # Fill both buffers
        for i in range(self.batch_size):
            # Positive rewards to win buffer
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            win_exp = Experience(state, torch.tensor(0), torch.tensor(1.0), state, torch.tensor(False))
            self.buffer.store(win_exp)
            
            # Zero rewards to other buffer
            other_exp = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(other_exp)
        
        # Sample batch
        states, actions, rewards, next_states, done = self.buffer.sample_batch()
        
        # Check shape
        self.assertEqual(states.shape[0], self.batch_size)
        self.assertEqual(rewards.shape[0], self.batch_size)
        
        # Check that we have both positive and zero rewards
        positive_count = (rewards > 0).sum().item()
        zero_count = (rewards == 0).sum().item()
        
        self.assertGreater(positive_count, 0)
        self.assertGreater(zero_count, 0)
        self.assertEqual(positive_count + zero_count, self.batch_size)
        
    def test_total_length(self):
        """Test total length across both buffers."""
        # Add to both buffers
        for i in range(5):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            win_exp = Experience(state, torch.tensor(0), torch.tensor(1.0), state, torch.tensor(False))
            other_exp = Experience(state, torch.tensor(0), torch.tensor(0.0), state, torch.tensor(False))
            self.buffer.store(win_exp)
            self.buffer.store(other_exp)
            
        # Total length should be 10
        self.assertEqual(len(self.buffer), 10)
        
    def test_cnt_rewards_total(self):
        """Test total reward count across both buffers."""
        # Add positive rewards
        for i in range(3):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            win_exp = Experience(state, torch.tensor(0), torch.tensor(1.0), state, torch.tensor(False))
            self.buffer.store(win_exp)
            
        # Add mixed rewards
        for i in range(2):
            state = torch.zeros(self.state_shape, dtype=torch.uint8)
            other_exp = Experience(state, torch.tensor(0), torch.tensor(0.5), state, torch.tensor(False))
            self.buffer.store(other_exp)
            
        # Total positive rewards should be 5
        self.assertEqual(self.buffer.cnt_rewards, 5)


if __name__ == '__main__':
    unittest.main()
