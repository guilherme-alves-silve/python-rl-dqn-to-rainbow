#!/usr/bin/env python
# coding: utf-8

"""
Deep Q-Network Training Script for Atari Pong

This script demonstrates how to use the refactored DQN modules to train
an agent to play Atari Pong using Deep Reinforcement Learning.
"""

import os
import sys
import logging
import traceback
import torch
import torch.nn as nn
import gymnasium as gym
import ale_py

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import refactored modules
from dqn_refactored import (
    AgentDQN,
    ReplayBuffer,
    BinaryReplayBuffer,
    PreprocessingWrapper,
    map_env_action,
    map_model_action,
    map_env2model,
    set_config_seed,
    make_wrapped_env
)

# Register Pong environment
gym.register_envs(ale_py)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def create_wrapped_env(env_name="PongNoFrameskip-v4", render_mode="rgb_array", **kwargs):
    """
    Create a wrapped environment with preprocessing.
    
    Args:
        env_name: Name of the Gymnasium environment
        render_mode: Render mode for visualization
        **kwargs: Additional environment parameters
        
    Returns:
        Preprocessed environment
    """
    env = gym.make(env_name, render_mode=render_mode, **kwargs)
    env.reset()
    env = PreprocessingWrapper(env)
    return env


def train_dqn_agent(
    env_name="PongNoFrameskip-v4",
    episodes=5000,
    epsilon=0.995,
    epsilon_min=0.1,
    epsilon_decay=0.9995,
    gamma=0.99,
    learning_rate=0.00025,
    update_interval=65,
    checkpoint_min_episodes=500,
    batch_size=128,
    buffer_size=50000,
    interactive=False,
    fill_percent=20,
    log_interval=500,
    load_checkpoint_episode=None,
    use_binary_buffer=False
):
    """
    Train a DQN agent on the specified environment.
    
    Args:
        env_name: Environment name
        episodes: Number of training episodes
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Exploration rate decay factor
        gamma: Discount factor for future rewards
        learning_rate: Learning rate for optimizer
        update_interval: Episodes between target network updates
        checkpoint_min_episodes: Minimum episodes before checkpointing
        batch_size: Batch size for training
        buffer_size: Replay buffer size
        interactive: Whether to show interactive plots
        fill_percent: Percentage of buffer to fill before training
        log_interval: Episodes between logging
        load_checkpoint_episode: Episode to resume from (None for fresh start)
        use_binary_buffer: Whether to use BinaryReplayBuffer
        
    Returns:
        Trained agent and replay buffer
    """
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = 42
    set_config_seed(seed)
    
    try:
        # Create environment
        env = create_wrapped_env(env_name)
        state, _ = env.reset()
        
        # Create agent
        agent = AgentDQN(
            device=device,
            episodes=episodes,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
            gamma=gamma,
            update_interval=update_interval,
            checkpoint_min_episodes=checkpoint_min_episodes,
            action_space_n=3,  # Pong has 3 actions after mapping
            learning_rate=learning_rate,
            interactive=interactive,
            map_env_action=map_env_action,
            map_model_action=map_model_action
        )
        
        # Create replay buffer
        if use_binary_buffer:
            replay_buffer = BinaryReplayBuffer(
                size=buffer_size,
                state_shape=state.shape,
                batch_size=batch_size,
                device=device,
                map_env2model=map_env2model
            )
        else:
            replay_buffer = ReplayBuffer(
                size=buffer_size,
                state_shape=state.shape,
                batch_size=batch_size,
                device=device,
                map_env2model=map_env2model
            )
        
        # Fill replay buffer with random experiences
        if fill_percent > 0:
            print(f"Filling replay buffer to {fill_percent}%...")
            agent.fill(env, replay_buffer, percent=fill_percent)
            
            # Save initial buffer
            buffer_path = f"replay_buffer_{'binary' if use_binary_buffer else 'standard'}_start.pth"
            replay_buffer.save(buffer_path)
            print(f"Saved initial buffer to {buffer_path}")
        
        # Train agent
        print("Starting training...")
        loss_func = nn.MSELoss()
        
        agent.train(
            env=env,
            log_interval=log_interval,
            epsilon=epsilon,
            replay_buffer=replay_buffer,
            loss_func=loss_func,
            start_episode=load_checkpoint_episode
        )
        
        # Save final buffer
        buffer_path = f"replay_buffer_{'binary' if use_binary_buffer else 'standard'}_end.pth"
        replay_buffer.save(buffer_path)
        print(f"Saved final buffer to {buffer_path}")
        
        # Print statistics
        print(f"Training completed!")
        print(f"Total positive rewards in buffer: {replay_buffer.cnt_rewards}")
        
        return agent, replay_buffer
        
    except Exception as ex:
        print(f"Training failed: {ex}")
        traceback.print_exc()
        raise


def evaluate_agent(agent, env_name="PongNoFrameskip-v4", episodes=3, epsilon=0.01, video_folder=None):
    """
    Evaluate a trained agent.
    
    Args:
        agent: Trained AgentDQN instance
        env_name: Environment name
        episodes: Number of episodes to evaluate
        epsilon: Exploration rate for evaluation (usually low)
        video_folder: Folder to save evaluation videos
        
    Returns:
        List of episode rewards
    """
    print(f"Evaluating agent for {episodes} episodes...")
    
    # Create environment
    env = create_wrapped_env(env_name)
    
    # Run evaluation
    rewards = agent.run(
        env=env,
        episodes=episodes,
        output=False,
        epsilon=epsilon,
        video_folder=video_folder
    )

    print(f"Evaluation results:")
    for i, reward in enumerate(rewards):
        print(f"  Episode {i+1}: {reward:.2f}")
    
    print(f"Average reward: {np.mean(rewards):.2f}")
    
    return rewards


def main():
    """Main function to run training and evaluation."""
    
    # Configuration
    config = {
        "env_name": "PongNoFrameskip-v4",
        "episodes": 5000,
        "epsilon": 0.995,
        "epsilon_min": 0.1,
        "epsilon_decay": 0.9995,
        "gamma": 0.99,
        "learning_rate": 0.00025,
        "update_interval": 65,
        "checkpoint_min_episodes": 500,
        "batch_size": 128,
        "buffer_size": 50000,
        "interactive": False,
        "fill_percent": 20,
        "log_interval": 500,
        "use_binary_buffer": False  # Change to True to use BinaryReplayBuffer
    }
    
    print("=" * 60)
    print("DQN Training for Atari Pong")
    print("=" * 60)
    print(f"Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print("=" * 60)

    agent, replay_buffer = train_dqn_agent(**config)

    print("\n" + "=" * 60)
    print("Evaluating trained agent...")
    print("=" * 60)
    
    evaluate_agent(
        agent=agent,
        env_name=config["env_name"],
        episodes=3,
        epsilon=0.01,
        video_folder="./evaluation_videos"
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
