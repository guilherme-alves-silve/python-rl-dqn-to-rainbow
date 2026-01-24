#!/usr/bin/env python
# coding: utf-8

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from network import Network


class TestNetwork(unittest.TestCase):
    """Unit tests for the Network class."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device("cpu")  # Use CPU for testing
        self.batch_size = 32
        self.input_channels = 4
        self.height = 84
        self.width = 84
        self.num_actions = 3
        
        # Create network instance
        self.network = Network(input_size=self.input_channels, actions=self.num_actions)
        
    def test_network_initialization(self):
        """Test that network is properly initialized."""
        self.assertIsInstance(self.network, nn.Module)
        self.assertIsInstance(self.network.conv1, nn.Conv2d)
        self.assertIsInstance(self.network.conv2, nn.Conv2d)
        self.assertIsInstance(self.network.conv3, nn.Conv2d)
        self.assertIsInstance(self.network.linear1, nn.LazyLinear)
        self.assertIsInstance(self.network.linear2, nn.LazyLinear)
        
        # Check layer parameters
        self.assertEqual(self.network.conv1.in_channels, self.input_channels)
        self.assertEqual(self.network.conv1.out_channels, 32)
        self.assertEqual(self.network.conv2.in_channels, 32)
        self.assertEqual(self.network.conv2.out_channels, 64)
        self.assertEqual(self.network.conv3.in_channels, 64)
        self.assertEqual(self.network.conv3.out_channels, 64)
        self.assertEqual(self.network.linear1.out_features, 512)
        self.assertEqual(self.network.linear2.out_features, self.num_actions)
        
    def test_forward_pass_single_batch(self):
        """Test forward pass with single sample."""
        # Create input tensor [1, 4, 84, 84]
        input_tensor = torch.randn(1, self.input_channels, self.height, self.width)
        
        # Forward pass
        output = self.network(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (1, self.num_actions))
        
        # Check output is not NaN or inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_forward_pass_batch(self):
        """Test forward pass with batch of samples."""
        # Create input tensor [32, 4, 84, 84]
        input_tensor = torch.randn(self.batch_size, self.input_channels, self.height, self.width)
        
        # Forward pass
        output = self.network(input_tensor)
        
        # Check output shape
        self.assertEqual(output.shape, (self.batch_size, self.num_actions))
        
        # Check output is not NaN or inf
        self.assertFalse(torch.isnan(output).any())
        self.assertFalse(torch.isinf(output).any())
        
    def test_forward_pass_different_sizes(self):
        """Test forward pass with different input sizes."""
        # Test with different batch sizes
        batch_sizes = [1, 2, 8, 16, 32]
        
        for batch_size in batch_sizes:
            input_tensor = torch.randn(batch_size, self.input_channels, self.height, self.width)
            output = self.network(input_tensor)
            self.assertEqual(output.shape, (batch_size, self.num_actions))
            
    def test_network_parameters(self):
        """Test that network has trainable parameters."""
        # Count parameters
        total_params = sum(p.numel() for p in self.network.parameters())
        trainable_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        
        # Should have parameters
        self.assertGreater(total_params, 0)
        self.assertGreater(trainable_params, 0)
        self.assertEqual(total_params, trainable_params)  # All should be trainable by default
        
    def test_relu_activations(self):
        """Test that ReLU activations are working."""
        # Create input with negative values
        input_tensor = torch.full((1, self.input_channels, self.height, self.width), -1.0)
        
        # Check that conv1 output has negative values before ReLU
        conv1_output = self.network.conv1(input_tensor)
        self.assertTrue((conv1_output < 0).any())
        
        # Check that ReLU output has no negative values (except possibly very small due to numerical precision)
        relu1_output = self.network.relu1(conv1_output)
        self.assertTrue((relu1_output >= 0).all())
        
    def test_gradient_flow(self):
        """Test that gradients flow through the network."""
        # Create input and target
        input_tensor = torch.randn(1, self.input_channels, self.height, self.width)
        target = torch.zeros(1, self.num_actions)
        target[0, 0] = 1.0  # Set first action as target
        
        # Forward pass
        output = self.network(input_tensor)
        
        # Calculate loss
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertFalse(torch.isnan(param.grad).any())
                self.assertFalse(torch.isinf(param.grad).any())
                
    def test_output_range(self):
        """Test that output can cover a reasonable range of Q-values."""
        # Test with various inputs
        test_cases = [
            torch.zeros(1, self.input_channels, self.height, self.width),
            torch.ones(1, self.input_channels, self.height, self.width),
            torch.randn(1, self.input_channels, self.height, self.width),
        ]
        
        for input_tensor in test_cases:
            output = self.network(input_tensor)
            
            # Output should not be all the same (unless input is all zeros)
            if not torch.allclose(input_tensor, torch.zeros_like(input_tensor)):
                self.assertFalse(torch.allclose(output[:, 0], output[:, 1]))
                
    def test_device_compatibility(self):
        """Test that network works on different devices."""
        device = self.device  # CPU for testing
        
        # Move network to device
        network = self.network.to(device)
        
        # Create input on device
        input_tensor = torch.randn(2, self.input_channels, self.height, self.width, device=device)
        
        # Forward pass
        output = network(input_tensor)
        
        # Check output is on correct device
        self.assertEqual(output.device.type, device.type)
        self.assertEqual(output.shape, (2, self.num_actions))


if __name__ == '__main__':
    unittest.main()
