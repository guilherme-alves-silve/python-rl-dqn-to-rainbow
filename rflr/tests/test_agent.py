#!/usr/bin/env python
# coding: utf-8

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import AgentDQN
from replay_buffer import ReplayBuffer, Experience
from network import Network


class TestAgentDQN(unittest.TestCase):
    """Unit tests for the AgentDQN class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")
        self.episodes = 100
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.gamma = 0.99
        self.update_interval = 10
        self.checkpoint_min_episodes = 50
        self.action_space_n = 3
        self.learning_rate = 0.00025
        
        # Create agent instance
        self.agent = AgentDQN(
            device=self.device,
            episodes=self.episodes,
            epsilon_min=self.epsilon_min,
            epsilon_decay=self.epsilon_decay,
            gamma=self.gamma,
            update_interval=self.update_interval,
            checkpoint_min_episodes=self.checkpoint_min_episodes,
            action_space_n=self.action_space_n,
            learning_rate=self.learning_rate,
            interactive=False  # Disable plots for testing
        )
        
    def test_agent_initialization(self):
        """Test that agent is properly initialized."""
        self.assertEqual(self.agent.episodes, self.episodes)
        self.assertEqual(self.agent.epsilon_min, self.epsilon_min)
        self.assertEqual(self.agent.epsilon_decay, self.epsilon_decay)
        self.assertEqual(self.agent.gamma, self.gamma)
        self.assertEqual(self.agent.update_interval, self.update_interval)
        self.assertEqual(self.agent.checkpoint_min_episodes, self.checkpoint_min_episodes)
        self.assertEqual(self.agent.action_space_n, self.action_space_n)
        self.assertFalse(self.agent.inference)
        self.assertFalse(self.agent.interactive)
        
        # Check networks
        self.assertIsInstance(self.agent.q_online, Network)
        self.assertIsInstance(self.agent.q_target, Network)
        self.assertIsInstance(self.agent.optimizer, torch.optim.RMSprop)
        
        # Check optimizer learning rate
        self.assertEqual(self.agent.optimizer.param_groups[0]['lr'], self.learning_rate)
        
    def test_select_action_greedy(self):
        """Test action selection with greedy policy (epsilon = 0)."""
        # Create mock environment
        mock_env = Mock()
        mock_env.action_space.sample.return_value = 0
        
        # Create state
        state = torch.zeros(4, 84, 84, dtype=torch.uint8)
        
        # Test with epsilon = 0 (should always choose best action)
        epsilon = 0.0
        action = self.agent.select_action(mock_env, epsilon, state)
        
        # Should return a valid action
        self.assertIn(action, [0, 1, 2])  # Assuming model actions are mapped to these
        
    def test_select_action_random(self):
        """Test action selection with random policy (epsilon = 1)."""
        # Create mock environment
        mock_env = Mock()
        mock_env.action_space.sample.return_value = 2
        
        # Create state
        state = torch.zeros(4, 84, 84, dtype=torch.uint8)
        
        # Test with epsilon = 1 (should always choose random action)
        epsilon = 1.0
        action = self.agent.select_action(mock_env, epsilon, state)
        
        # Should return the mocked random action
        self.assertEqual(action, 2)
        
    def test_set_mode_train(self):
        """Test setting network to training mode."""
        # Create test network
        network = Network(input_size=4, actions=3)
        
        # Set to train mode
        self.agent._set_mode(network, freeze=False)
        
        # Check mode
        self.assertTrue(network.training)
        
        # Check that parameters require gradients
        for param in network.parameters():
            self.assertTrue(param.requires_grad)
            
    def test_set_mode_eval(self):
        """Test setting network to evaluation mode."""
        # Create test network
        network = Network(input_size=4, actions=3)
        
        # Set to eval mode
        self.agent._set_mode(network, freeze=True)
        
        # Check mode
        self.assertFalse(network.training)
        
        # Check that parameters don't require gradients
        for param in network.parameters():
            self.assertFalse(param.requires_grad)
            
    def test_reduce_epsilon(self):
        """Test epsilon decay."""
        # Test above minimum
        epsilon = 0.5
        new_epsilon = self.agent._reduce_epsilon(epsilon)
        expected = max(self.epsilon_min, epsilon * self.epsilon_decay)
        self.assertAlmostEqual(new_epsilon, expected)
        
        # Test at minimum
        epsilon = 0.05  # Below epsilon_min
        new_epsilon = self.agent._reduce_epsilon(epsilon)
        self.assertEqual(new_epsilon, self.epsilon_min)
        
    def test_update_q_target(self):
        """Test target network update."""
        # Modify online network weights
        with torch.no_grad():
            for param in self.agent.q_online.parameters():
                param.fill_(0.5)
                
        # Update target network
        self.agent._update_q_target(10)  # Episode divisible by update_interval
        
        # Check that target network has same weights as online
        for online_param, target_param in zip(self.agent.q_online.parameters(), 
                                              self.agent.q_target.parameters()):
            torch.testing.assert_close(target_param, online_param)
            
    def test_update_q_target_not_due(self):
        """Test that target network is not updated when not due."""
        # Store original target weights
        original_weights = []
        for param in self.agent.q_target.parameters():
            original_weights.append(param.clone())
            
        # Try to update at wrong episode
        self.agent._update_q_target(5)  # Episode not divisible by update_interval
        
        # Check that target weights didn't change
        for original, current in zip(original_weights, self.agent.q_target.parameters()):
            torch.testing.assert_close(current, original)
            
    def test_checkpoint_save_condition(self):
        """Test checkpoint saving conditions."""
        trackers = {
            "epsilon": [0.1],
            "reward": [100.0],
            "loss": [0.5]
        }
        
        # Test not enough episodes
        best_reward = self.agent._checkpoint(10, 50.0, 0.0, 1000, trackers)
        self.assertEqual(best_reward, 0.0)  # Shouldn't save
        
        # Test reward not better than best
        best_reward = self.agent._checkpoint(60, 50.0, 100.0, 1000, trackers)
        self.assertEqual(best_reward, 100.0)  # Shouldn't save
        
        # Test good conditions for saving
        best_reward = self.agent._checkpoint(60, 150.0, 100.0, 1000, trackers)
        self.assertEqual(best_reward, 150.0)  # Should save
        
    def test_save_and_load_checkpoint(self):
        """Test saving and loading checkpoints."""
        # Create some test data
        trackers = {
            "epsilon": [0.1],
            "reward": [100.0],
            "loss": [0.5]
        }
        
        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pth")
            
            # Save training state
            self.agent.save_training(50, 100.0, 5000, "test_checkpoint", trackers)
            
            # Check file exists
            self.assertTrue(os.path.exists(checkpoint_path))
            
            # Load checkpoint
            loaded_agent = AgentDQN.from_checkpoint(
                execution_id="test_load",
                episode=50,
                episodes=self.episodes,
                action_space_n=self.action_space_n,
                file_path=checkpoint_path,
                inference=True,
                interactive=False
            )
            
            # Check loaded agent properties
            self.assertEqual(loaded_agent.epsilon_min, self.epsilon_min)
            self.assertEqual(loaded_agent.epsilon_decay, self.epsilon_decay)
            self.assertEqual(loaded_agent.gamma, self.gamma)
            self.assertTrue(loaded_agent.inference)
            
    def test_train_q_online_not_enough_samples(self):
        """Test training when buffer doesn't have enough samples."""
        # Create small buffer
        buffer = ReplayBuffer(
            size=100,
            state_shape=torch.Size([4, 84, 84]),
            batch_size=32,
            device=self.device,
            map_env2model=lambda x: x
        )
        
        # Try to train with insufficient samples
        loss = self.agent._train_q_online(buffer, nn.MSELoss())
        
        # Should return None
        self.assertIsNone(loss)
        
    def test_update_q_target_weights(self):
        """Test that target network weights are properly updated."""
        # Set online network to known values
        with torch.no_grad():
            for param in self.agent.q_online.parameters():
                param.fill_(1.0)
                
        # Set target network to different values
        with torch.no_grad():
            for param in self.agent.q_target.parameters():
                param.fill_(0.0)
                
        # Update target network
        self.agent._update_q_target(10)
        
        # Check that target now matches online
        for online_param, target_param in zip(self.agent.q_online.parameters(), 
                                              self.agent.q_target.parameters()):
            torch.testing.assert_close(target_param, online_param)
            
    def test_network_modes(self):
        """Test network modes (train vs eval)."""
        # Check initial modes
        self.assertTrue(self.agent.q_online.training)
        self.assertFalse(self.agent.q_target.training)
        
        # Check parameter requirements
        for param in self.agent.q_online.parameters():
            self.assertTrue(param.requires_grad)
            
        for param in self.agent.q_target.parameters():
            self.assertFalse(param.requires_grad)


if __name__ == '__main__':
    unittest.main()
