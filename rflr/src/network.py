#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn


class Network(nn.Module):
    """
    Deep Q-Network architecture based on the Nature paper:
    "Human-level Control Through Deep Reinforcement Learning"
    
    The network consists of 3 convolutional layers followed by 2 fully connected layers.
    It takes preprocessed frames as input and outputs Q-values for each action.
    """

    def __init__(self, input_size: int, actions: int):
        """
        Initialize the Deep Q-Network.
        
        Args:
            input_size: Number of input channels (typically 4 for frame stacking)
            actions: Number of possible actions
        """
        super(Network, self).__init__()
        
        # First convolutional layer: 32 filters of 8x8 with stride 4
        self.conv1 = nn.Conv2d(in_channels=input_size, out_channels=32, stride=4, kernel_size=(8, 8))
        self.relu1 = nn.ReLU()
        
        # Second convolutional layer: 64 filters of 4x4 with stride 2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=(4, 4))
        self.relu2 = nn.ReLU()
        
        # Third convolutional layer: 64 filters of 3x3 with stride 1
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, stride=1, kernel_size=(3, 3))
        self.relu3 = nn.ReLU()
        
        # Flatten layer to convert from conv to linear
        self.flatten = nn.Flatten()
        
        # Fully connected layer with 512 units
        # LazyLinear calculates the input size automatically
        self.linear1 = nn.LazyLinear(out_features=512)
        self.relu4 = nn.ReLU()
        
        # Output layer: Q-values for each action (no activation)
        self.linear2 = nn.LazyLinear(actions)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Q-values for each action: shape [batch_size, num_actions]
        """
        # Convolutional layers with ReLU activations
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        
        # Flatten and pass through fully connected layers
        x = self.flatten(x)
        x = self.relu4(self.linear1(x))
        x = self.linear2(x)
        
        return x
