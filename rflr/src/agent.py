#!/usr/bin/env python
# coding: utf-8

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium import Env
from tqdm import tqdm
from typing import Tuple, Self
from uuid import uuid4

from .network import Network
from .replay_buffer import ReplayBuffer, Experience
from .utils import timeit


class AgentDQN:
    """
    Deep Q-Network Agent for playing Atari games.

    Implements the DQN algorithm with experience replay and target networks
    as described in "Human-level Control Through Deep Reinforcement Learning".
    """

    def __init__(self,
                 device: torch.device,
                 episodes: int,
                 epsilon_min: float,
                 epsilon_decay: float,
                 gamma: float,
                 update_interval: int,
                 checkpoint_min_episodes: int,
                 action_space_n: int,
                 learning_rate: float = 0.00025,
                 cat_input_size: int = 4,
                 inference: bool = False,
                 interactive: bool = True,
                 execution_id: str = None,
                 map_env_action=lambda action: action,
                 map_model_action=lambda model_action: model_action,
                 _loading_checkpoint: bool = False):
        """
        Initialize the DQN Agent.

        Args:
            device: PyTorch device (CPU/GPU)
            episodes: Number of training episodes
            epsilon_min: Minimum exploration rate
            epsilon_decay: Decay factor for exploration rate
            gamma: Discount factor for future rewards
            update_interval: Episodes between target network updates
            checkpoint_min_episodes: Minimum episodes before saving checkpoints
            action_space_n: Number of possible actions
            learning_rate: Learning rate for optimizer
            cat_input_size: Number of concatenated frames
            inference: Whether to run in inference mode
            interactive: Whether to show interactive plots
            execution_id: Unique identifier for this run
            map_env_action: Function to map environment actions
            map_model_action: Function to map model actions
            _loading_checkpoint: Internal flag for checkpoint loading
        """
        self.execution_id = uuid4().hex if execution_id is None else execution_id
        self.device = device
        self.episodes = episodes
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.update_interval = update_interval
        self.checkpoint_min_episodes = checkpoint_min_episodes
        self.inference = inference
        self.interactive = interactive
        self.map_env_action = map_env_action
        self.map_model_action = map_model_action
        self.cnt_frames = 0
        self.logger = logging.getLogger(self.__class__.__name__)

        if not _loading_checkpoint:
            q_online = Network(input_size=cat_input_size, 
                               actions=action_space_n)
            q_target = Network(input_size=cat_input_size, 
                               actions=action_space_n)

            q_target.load_state_dict(q_online.state_dict())

            self._set_mode(q_online)
            self._set_mode(q_target, freeze=True)

            self.q_online = q_online.to(device)
            self.q_target = q_target.to(device)

            self.optimizer = optim.RMSprop(q_online.parameters(), 
                                           lr=learning_rate)

            os.makedirs(f"./checkpoints_{self.execution_id}", exist_ok=True)
            self.logger.info(f"Created the agent: {self.execution_id}")
        else:
            self.q_online = None
            self.q_target = None
            self.optimizer = None
            self.logger.info(f"Loading checkpoint...")

    @torch.no_grad()
    def select_action(self, env: Env, epsilon: float, state: torch.Tensor):
        """
        Select action using epsilon-greedy policy.
        
        With probability epsilon, choose a random action (exploration).
        Otherwise, choose the action with highest Q-value (exploitation).
        
        Args:
            env: Gymnasium environment
            epsilon: Exploration rate
            state: Current state tensor
            
        Returns:
            Selected action
        """
        # Mr. Krabs, remember? epsilon-greedy
        if torch.rand(1).item() < epsilon:
            return self.map_env_action(env.action_space.sample())

        # Preprocess state for network input
        # [C, H, W] -> [1, C, H, W]
        state = (state.to(self.device, dtype=torch.float32)
                      .div_(255.)
                      .unsqueeze_(dim=0))

        action = self.q_online(state).argmax(dim=1)
        return self.map_model_action(action.item())

    def _get_fig_axes(self, plot_trackers_count: int) -> Tuple[plt.Figure, list[plt.Axes]]:
        """
        Create figure and axes for plotting training progress.

        Args:
            plot_trackers_count: Number of metrics to plot

        Returns:
            Tuple of (figure, axes_list)
        """
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6 * plot_trackers_count, 5))
        axs = fig.subplots(nrows=1, 
                           ncols=plot_trackers_count)

        if not isinstance(axs, (list, tuple, np.ndarray)):
            axs = [axs]

        if self.interactive:
            plt.ion()
            plt.show(block=False)

        return fig, axs

    def _draw_progress(self,
                       pbar: tqdm,
                       fig: plt.Figure,
                       axs: list[plt.Axes],
                       episode: int, 
                       log_interval: int, 
                       cnt_frames: int,
                       episode_tracker: list[int],
                       plot_trackers: dict[str, list]) -> None:
        """
        Update progress bar and training plots.

        Args:
            pbar: Progress bar instance
            fig: Figure for plotting
            axs: List of axes for different metrics
            episode: Current episode number
            log_interval: Episodes between logging
            cnt_frames: Total frames processed
            episode_tracker: List of episode numbers
            plot_trackers: Dictionary of metric lists
        """
        import matplotlib.pyplot as plt
        from IPython.display import display, clear_output

        if episode == 0:
            return

        pbar.set_postfix({
            "ε": f"{plot_trackers['epsilon'][-1]:.4f}",
            "🪙": f"{plot_trackers['reward'][-1]:.4f}",
            "📉": f"{plot_trackers['loss'][-1]:.4f}",
            "🖼️": str(cnt_frames)
        })

        # Update plots at intervals
        if (episode % log_interval == 0 or
            episode == self.episodes):
            reward = int(plot_trackers["reward"][-1])
            self.logger.info(f"Rewards: {reward}, count frames: {cnt_frames}")

            for ax, (title, tracker) in zip(axs, plot_trackers.items()):
                ax.clear()
                ax.set_title(title)
                ax.plot(episode_tracker, tracker)
            plt.tight_layout()
            fig.canvas.draw_idle()
            if not self.interactive:
                display(fig)

    def _set_mode(self, model: Network, freeze: bool = False) -> None:
        """
        Set network mode (train/eval) and freeze parameters if needed.
        
        Args:
            model: Network to configure
            freeze: Whether to freeze parameters (no gradients)
        """
        if freeze or self.inference:
            # Don't compute statistics
            model.eval()
            # Don't compute gradients
            for param in model.parameters():
                param.requires_grad = False
        else:
            model.train()

    def _checkpoint(self,
                    episode: int,
                    current_total_reward: float,
                    best_total_reward: float,
                    cnt_frames: int,
                    trackers: dict,
                    delta: float = 0.0001) -> float:
        """
        Save checkpoint if current performance is better than best.

        Args:
            episode: Current episode
            current_total_reward: Reward from current episode
            best_total_reward: Best reward seen so far
            cnt_frames: Total frames processed
            trackers: Training metrics
            delta: Small value to avoid floating point comparison issues

        Returns:
            Updated best reward
        """

        if (episode <= 0 or
            current_total_reward < (best_total_reward + delta) or
            episode < self.checkpoint_min_episodes):
            return best_total_reward

        self.logger.info(f"Creating checkpoint because current_total_reward > best_total_reward: "
                         f"{current_total_reward} > {best_total_reward}")
        self.save_training(episode,
                           current_total_reward,
                           cnt_frames,
                           f"checkpoint_{str(episode).replace('.', '_')}_{current_total_reward}",
                           trackers)
        return current_total_reward

    def save_training(self,
                      episode: int,
                      reward: float,
                      cnt_frames: int,
                      file_name: str,
                      trackers: dict = None) -> None:
        """
        Save training checkpoint to disk.

        Args:
            episode: Current episode
            reward: Current episode reward
            cnt_frames: Total frames processed
            file_name: Name for checkpoint file
            trackers: Training metrics
        """
        if not trackers:
            trackers = {
                "epsilon": [np.nan],
                "reward": [reward],
                "loss": [np.nan],
            }

        checkpoint_info = {
            "episode": episode,
            "cnt_frames": cnt_frames,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "gamma": self.gamma,
            "update_interval": self.update_interval,
            "checkpoint_min_episodes": self.checkpoint_min_episodes,
            "learning_rate": self.optimizer.param_groups[0]['lr'],
            "model_state_dict": self.q_online.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": trackers["epsilon"][-1],
            "reward": trackers["reward"][-1],
            "loss": trackers["loss"][-1],
        }

        torch.save(checkpoint_info, f"./checkpoints_{self.execution_id}/{file_name}.pth")

    @classmethod
    def from_checkpoint(cls, 
                        execution_id: str, 
                        episode: int,
                        episodes: int,
                        action_space_n: int,
                        file_path: str,
                        cat_input_size: int = 4,
                        inference: bool = False,
                        interactive: bool = False,
                        map_env_action=lambda action: action) -> Self:
        """
        Load agent from checkpoint file.
        
        Args:
            execution_id: Unique run identifier
            episode: Episode to resume from
            episodes: Total episodes to train
            action_space_n: Number of actions
            file_path: Path to checkpoint file
            cat_input_size: Number of concatenated frames
            inference: Whether to run in inference mode
            interactive: Whether to show interactive plots
            map_env_action: Function to map environment actions
            
        Returns:
            Loaded AgentDQN instance
        """
        logging.getLogger(cls.__name__).debug(f"Getting agent from checkpoint: {file_path}")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"No checkpoint file \"{file_path}\" found for execution_id: {execution_id}")

        checkpoint_info = torch.load(file_path,
                                     weights_only=False,
                                     map_location="cpu")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent = cls(
            device=device,
            episodes=episodes,
            epsilon_min=checkpoint_info["epsilon_min"],
            epsilon_decay=checkpoint_info["epsilon_decay"],
            gamma=checkpoint_info["gamma"],
            update_interval=checkpoint_info["update_interval"],
            checkpoint_min_episodes=checkpoint_info["checkpoint_min_episodes"],
            action_space_n=action_space_n,
            cat_input_size=cat_input_size,
            inference=inference,
            interactive=interactive,
            execution_id=execution_id,
            map_env_action=map_env_action,
            _loading_checkpoint=True
        )

        q_online = Network(input_size=cat_input_size,
                           actions=action_space_n)
        q_target = Network(input_size=cat_input_size,
                           actions=action_space_n)
        q_online.load_state_dict(checkpoint_info["model_state_dict"])
        q_target.load_state_dict(checkpoint_info["model_state_dict"])

        agent._set_mode(q_online, freeze=inference)
        agent._set_mode(q_target, freeze=True)

        agent.q_online = q_online.to(device)
        agent.q_target = q_target.to(device)

        agent.optimizer = optim.RMSprop(agent.q_online.parameters(),
                                        lr=checkpoint_info["learning_rate"])
        agent.optimizer.load_state_dict(checkpoint_info["optimizer_state_dict"])

        agent.logger.info(f"Loaded agent from checkpoint: {file_path}")
        agent.logger.info(f"Checkpoint episode: {checkpoint_info['episode']}")
        agent.logger.info(f"Checkpoint epsilon: {checkpoint_info['epsilon']:.4f}")
        agent.logger.info(f"Checkpoint reward: {checkpoint_info['reward']:.4f}")
        agent.logger.info(f"Checkpoint loss: {checkpoint_info['loss']:.4f}")

        return agent

    @timeit
    def train(self,
              env: Env,
              log_interval: int,
              epsilon: float,
              replay_buffer: ReplayBuffer,
              loss_func: nn.Module,
              start_episode: int = 0) -> None:
        """
        Main training loop for the DQN agent.

        Args:
            env: Gymnasium environment
            log_interval: Episodes between logging
            epsilon: Initial exploration rate
            replay_buffer: Experience replay buffer
            loss_func: Loss function (e.g., MSELoss)
            start_episode: Episode to resume from (for checkpoint loading)
        """
        import matplotlib.pyplot as plt
        
        self.logger.info("Training the agent...")

        start_episode = start_episode if start_episode else 0
        best_reward = float('-inf')
        episode_tracker = []
        trackers = {
            "epsilon": [],
            "reward": [],
            "loss": []
        }

        cnt_frames = 0
        fig, axs = self._get_fig_axes(len(trackers))

        for episode in (pbar := tqdm(range(start_episode + 1, self.episodes),
                                     unit="episode",
                                     desc="Training",
                                     ncols=100,
                                     mininterval=5.0)):
            state, _ = env.reset()
            episode_rewards = []
            episode_loss = []

            while True:

                action = self.select_action(env, epsilon, state)
                next_state, reward, term, trunc, _ = env.step(action)
                done = term or trunc

                replay_buffer.store(Experience(state, action, reward, next_state, done))
                loss = self._train_q_online(replay_buffer, loss_func)

                episode_rewards.append(reward)
                episode_loss.append(loss if loss else np.nan)

                state = next_state
                cnt_frames += 1

                if done:
                    break

            self._update_q_target(episode)
            epsilon = self._reduce_epsilon(epsilon)

            episode_total_reward = np.sum(episode_rewards)
            episode_tracker.append(episode)
            trackers["epsilon"].append(epsilon)
            trackers["reward"].append(episode_total_reward)
            trackers["loss"].append(np.mean(episode_loss))

            best_reward = self._checkpoint(episode, episode_total_reward, best_reward, 
                                           cnt_frames, trackers)

            self._draw_progress(pbar, fig, axs, 
                                episode, log_interval, cnt_frames, 
                                episode_tracker=episode_tracker,
                                plot_trackers=trackers)

        env.close()
        self.logger.info("Training finished!")

    def _train_q_online(self,
                        replay_buffer: ReplayBuffer, 
                        loss_func: nn.Module) -> float:
        """
        Train the online Q-network using a batch from replay buffer.

        Implements the DQN loss:
        y = r + γ * max Q_target(s', a') * (1 - done)
        Loss = MSE(Q_online(s, a), y)

        Args:
            replay_buffer: Experience replay buffer
            loss_func: Loss function

        Returns:
            Loss value as float, or None if buffer doesn't have enough samples
        """
        if not replay_buffer.enough():
            return None

        self.logger.debug("Training Q-online:")

        states, actions, rewards, next_states, done = replay_buffer.sample_batch()

        # actions shape from [B] to [B, 1] then back to [B]
        yhat = self.q_online(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze(dim=1)

        with torch.no_grad():
            # y = r + γ * max Q_target(s', a') * (1 - done)
            y = rewards + self.gamma * self.q_target(next_states).max(dim=1).values * (1.0 - done.float())
        
        # Calculate loss
        loss = loss_func(yhat, y)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _update_q_target(self, episode: int) -> None:
        """
        Update target network with online network weights.
        
        This is done every C steps to stabilize training.
        
        Args:
            episode: Current episode number
        """
        if episode % self.update_interval != 0:
            return

        self.logger.debug(f"Updating Q-target in episode: {episode}")

        # Copy weights from online to target network
        self.q_target.load_state_dict(self.q_online.state_dict())
        self._set_mode(self.q_target, freeze=True)

    def _reduce_epsilon(self, epsilon: float):
        """
        Decay exploration rate using exponential decay.
        
        ε = max(ε_min, ε × ε_decay)
        
        Args:
            epsilon: Current exploration rate
            
        Returns:
            New exploration rate
        """
        return max(self.epsilon_min, epsilon * self.epsilon_decay)

    @timeit
    def run(self,
            env: Env,
            episodes: int = 10,
            output: bool = True,
            epsilon: float = 0.01,
            video_folder: str = None) -> list[float]:
        """
        Run the trained agent in the environment.
        
        Args:
            env: Gymnasium environment
            episodes: Number of episodes to run
            output: Whether to render environment
            epsilon: Exploration rate (usually low for evaluation)
            video_folder: Folder to save video recordings
            
        Returns:
            List of total rewards per episode
        """
        import matplotlib.pyplot as plt
        from IPython.display import clear_output, display
        
        if video_folder:
            self.logger.info("Recoding agent behavior!")
            env = gym.wrappers.RecordVideo(env, video_folder=video_folder)

        total_reward = 0
        episodes_total_reward = []
        
        for episode in range(episodes):
            state, _ = env.reset()
            
            while True:
                if output:
                    from .utils import show_environment
                    show_environment(env.render())
                
                # Select action and step environment
                action = self.select_action(env, epsilon, state)
                next_state, reward, term, trunc, _ = env.step(action)
                state = next_state

                total_reward += reward
                
                if term or trunc:
                    break
            
            self.logger.info(f"Total reward in episode {episode}: {total_reward}")
            episodes_total_reward.append(total_reward)
            total_reward = 0
        
        env.close()
        return episodes_total_reward

    @timeit
    def fill(self,
             env: Env,
             replay_buffer: ReplayBuffer,
             percent: int = 100) -> bool:
        """
        Fill replay buffer with random experiences before training.
        
        This helps prevent overfitting in early training.
        
        Args:
            env: Gymnasium environment
            replay_buffer: Replay buffer to fill
            percent: Percentage of buffer capacity to fill
            
        Returns:
            True if buffer was filled, False otherwise
        """
        assert percent >= 0, "percent must be >= 0"
        if percent == 0:
            self.logger.info("The replay buffer won't be filled!")
            return False

        target_count = int(replay_buffer.size * percent / 100)

        from tqdm import tqdm
        
        with tqdm(total=target_count,
                  desc=f"Filling replay buffer to {percent}%",
                  ncols=100,
                  mininterval=5.0) as pbar:
            filling = lambda: pbar.n < target_count

            while filling():
                state, _ = env.reset()
                while filling():
                    action = self.map_env_action(env.action_space.sample())
                    next_state, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    replay_buffer.store(Experience(state, action, reward, next_state, done))
                    pbar.update(1)
                    state = next_state
                    if done:
                        break
        
        env.close()
        return True
