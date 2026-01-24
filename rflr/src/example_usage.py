#!/usr/bin/env python
# coding: utf-8

"""
Example usage of the refactored DQN modules

This script demonstrates various ways to use the refactored DQN components.
"""

import torch
import torch.nn as nn
import gymnasium as gym
import ale_py
import sys
import os
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import refactored modules
from dqn_refactored import (
    AgentDQN,
    Network,
    ReplayBuffer,
    BinaryReplayBuffer,
    PreprocessingWrapper,
    Experience,
    map_env_action,
    map_model_action,
    map_env2model,
    set_config_seed,
    get_visual_array,
    debug_concat_frames
)

# Register Pong environment
gym.register_envs(ale_py)


def example_1_create_network():
    """Example: Create and test the DQN network."""
    print("Example 1: Creating and testing the DQN Network")
    print("-" * 50)
    
    # Create network
    network = Network(input_size=4, actions=3)
    
    # Create random input (batch of 32 states, 4 frames each, 84x84)
    input_tensor = torch.randn(32, 4, 84, 84)
    
    # Forward pass
    output = network(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
    
    # Test with different batch sizes
    for batch_size in [1, 8, 16]:
        test_input = torch.randn(batch_size, 4, 84, 84)
        test_output = network(test_input)
        assert test_output.shape == (batch_size, 3), f"Expected ({batch_size}, 3), got {test_output.shape}"
    
    print("✅ Network test passed!")
    print()


def example_2_replay_buffer():
    """Example: Using the replay buffer."""
    print("Example 2: Using the Replay Buffer")
    print("-" * 50)
    
    # Create replay buffer
    device = torch.device("cpu")
    buffer = ReplayBuffer(
        size=1000,
        state_shape=torch.Size([4, 84, 84]),
        batch_size=32,
        device=device,
        map_env2model=map_env2model
    )
    
    # Create some experiences
    for i in range(100):
        state = torch.randint(0, 255, (4, 84, 84), dtype=torch.uint8)
        action = torch.tensor(i % 3, dtype=torch.uint8)
        reward = torch.tensor(float(i % 10), dtype=torch.float32)
        next_state = torch.randint(0, 255, (4, 84, 84), dtype=torch.uint8)
        done = torch.tensor(i % 20 == 0, dtype=torch.bool)
        
        experience = Experience(state, action, reward, next_state, done)
        buffer.store(experience)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Has enough for batch: {buffer.enough()}")
    print(f"Positive rewards: {buffer.cnt_rewards}")
    
    # Sample a batch
    if buffer.enough():
        states, actions, rewards, next_states, done = buffer.sample_batch()
        print(f"Sampled batch - States: {states.shape}, Actions: {actions.shape}")
        print(f"States range: [{states.min():.3f}, {states.max():.3f}]")
    
    print("✅ Replay buffer test passed!")
    print()


def example_3_binary_replay_buffer():
    """Example: Using the binary replay buffer."""
    print("Example 3: Using the Binary Replay Buffer")
    print("-" * 50)
    
    # Create binary replay buffer
    device = torch.device("cpu")
    buffer = BinaryReplayBuffer(
        size=1000,
        state_shape=torch.Size([4, 84, 84]),
        batch_size=32,
        device=device,
        map_env2model=map_env2model
    )
    
    # Add experiences with mixed rewards
    for i in range(100):
        state = torch.randint(0, 255, (4, 84, 84), dtype=torch.uint8)
        action = torch.tensor(i % 3, dtype=torch.uint8)
        
        # Mix of positive and non-positive rewards
        if i % 3 == 0:
            reward = torch.tensor(1.0, dtype=torch.float32)  # Positive
        else:
            reward = torch.tensor(0.0, dtype=torch.float32)  # Non-positive
            
        next_state = torch.randint(0, 255, (4, 84, 84), dtype=torch.uint8)
        done = torch.tensor(False, dtype=torch.bool)
        
        experience = Experience(state, action, reward, next_state, done)
        buffer.store(experience)
    
    print(f"Total buffer size: {len(buffer)}")
    print(f"Win buffer size: {len(buffer.win_replay_buffer)}")
    print(f"Other buffer size: {len(buffer.other_replay_buffer)}")
    print(f"Has enough in both: {buffer.enough()}")
    
    if buffer.enough():
        states, actions, rewards, next_states, done = buffer.sample_batch()
        positive_count = (rewards > 0).sum().item()
        print(f"Sampled batch - Positive rewards: {positive_count}/32")
    
    print("✅ Binary replay buffer test passed!")
    print()


def example_4_preprocessing():
    """Example: Using the preprocessing wrapper."""
    print("Example 4: Using the Preprocessing Wrapper")
    print("-" * 50)
    
    # Create a simple mock environment for testing
    class MockEnv:
        def __init__(self):
            self.observation_space = gym.spaces.Box(low=0, high=255, shape=(210, 160, 3), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(6)
            self.step_count = 0
            
        def reset(self, seed=None, options=None):
            state = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
            return state, {}
            
        def step(self, action):
            self.step_count += 1
            state = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
            reward = 1.0
            term = self.step_count >= 10
            trunc = False
            info = {}
            return state, reward, term, trunc, info
    
    # Create mock environment and wrapper
    mock_env = MockEnv()
    wrapper = PreprocessingWrapper(env=mock_env)
    
    # Test reset
    state, info = wrapper.reset()
    print(f"Reset state shape: {state.shape}")
    print(f"Reset state dtype: {state.dtype}")
    
    # Test step
    next_state, reward, term, trunc, info = wrapper.step(0)
    print(f"Next state shape: {next_state.shape}")
    print(f"Next state dtype: {next_state.dtype}")
    print(f"Reward: {reward}")
    
    # Test individual preprocessing steps
    test_frame = np.random.randint(0, 255, (210, 160, 3), dtype=np.uint8)
    
    # Grayscale conversion
    gray_tensor = wrapper._grayscale_frame(test_frame)
    print(f"Grayscale shape: {gray_tensor.shape}")
    
    # Resize
    resized = wrapper._resize_frame(gray_tensor)
    print(f"Resized shape: {resized.shape}")
    
    print("✅ Preprocessing test passed!")
    print()


def example_5_action_mapping():
    """Example: Action mapping functions."""
    print("Example 5: Action Mapping Functions")
    print("-" * 50)
    
    # Test environment to model mapping
    env_actions = [0, 2, 3]
    print("Environment to Model mapping:")
    for action in env_actions:
        model_action = map_env2model(action)
        print(f"  env_action {action} -> model_action {model_action}")
    
    # Test model to environment mapping
    model_actions = [0, 1, 2]
    print("\nModel to Environment mapping:")
    for action in model_actions:
        env_action = map_model_action(action)
        print(f"  model_action {action} -> env_action {env_action}")
    
    # Test environment action mapping
    print("\nEnvironment action mapping (6 -> 3 actions):")
    for action in range(6):
        try:
            mapped = map_env_action(action)
            print(f"  env_action {action} -> {mapped}")
        except ValueError:
            print(f"  env_action {action} -> Invalid")
    
    print("✅ Action mapping test passed!")
    print()


def example_6_utility_functions():
    """Example: Utility functions."""
    print("Example 6: Utility Functions")
    print("-" * 50)
    
    # Test seed setting
    print("Testing seed setting...")
    set_config_seed(42)
    val1 = torch.rand(1).item()
    
    set_config_seed(42)
    val2 = torch.rand(1).item()
    
    print(f"Same seed produces same value: {val1 == val2}")
    
    # Test visual array function
    tensor = torch.randn(4, 84, 84)
    array, cmap = get_visual_array(tensor)
    print(f"Visual array shape: {array.shape}")
    print(f"Colormap: {cmap}")
    
    print("✅ Utility functions test passed!")
    print()


def example_7_create_agent():
    """Example: Creating and configuring a DQN agent."""
    print("Example 7: Creating a DQN Agent")
    print("-" * 50)
    
    # Set device
    device = torch.device("cpu")  # Use CPU for example
    
    # Create agent configuration
    agent_config = {
        "device": device,
        "episodes": 1000,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.995,
        "gamma": 0.99,
        "update_interval": 10,
        "checkpoint_min_episodes": 100,
        "action_space_n": 3,
        "learning_rate": 0.00025,
        "interactive": False,
        "map_env_action": map_env_action,
        "map_model_action": map_model_action
    }
    
    # Create agent
    agent = AgentDQN(**agent_config)
    
    print(f"Agent created with:")
    print(f"  Episodes: {agent.episodes}")
    print(f"  Gamma: {agent.gamma}")
    print(f"  Epsilon decay: {agent.epsilon_decay}")
    print(f"  Update interval: {agent.update_interval}")
    print(f"  Device: {agent.device}")
    
    # Check networks
    print(f"  Online network: {type(agent.q_online).__name__}")
    print(f"  Target network: {type(agent.q_target).__name__}")
    print(f"  Optimizer: {type(agent.optimizer).__name__}")
    
    print("✅ Agent creation test passed!")
    print()


def main():
    """Run all examples."""
    print("=" * 60)
    print("DQN Refactored Modules - Example Usage")
    print("=" * 60)
    print()
    
    # Run all examples
    example_1_create_network()
    example_2_replay_buffer()
    example_3_binary_replay_buffer()
    example_4_preprocessing()
    example_5_action_mapping()
    example_6_utility_functions()
    example_7_create_agent()
    
    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("1. Run the unit tests: python -m tests.run_tests")
    print("2. Train an agent: python train_pong.py")
    print("3. Check the README.md for detailed documentation")


if __name__ == "__main__":
    main()
