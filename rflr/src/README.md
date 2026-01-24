# Deep Q-Network Implementation - Refactored

This is a refactored implementation of Deep Q-Network (DQN) for playing Atari Pong, based on the original code. 
The refactoring separates concerns into modular components and adds comprehensive unit tests.

## Project Structure

```
src/
├── __init__.py              # Package initialization
├── agent.py                 # AgentDQN class with all methods
├── network.py              # Deep Q-Network architecture
├── replay_buffer.py        # Experience replay buffers
├── preprocessing.py        # Environment preprocessing wrapper
├── utils.py               # Utility functions and helpers
├── train_pong.py          # Main training script
├── tests/                 # Unit tests
│   ├── __init__.py
│   ├── test_agent.py      # Tests for AgentDQN
│   ├── test_network.py    # Tests for Network
│   ├── test_replay_buffer.py  # Tests for replay buffers
│   ├── test_preprocessing.py  # Tests for preprocessing
│   ├── test_utils.py      # Tests for utilities
│   └── run_tests.py       # Test runner
└── README.md              # This file
```

## Key Improvements

### 1. **Modular Architecture**
- Each class is in its own file for better organization
- Clear separation of concerns
- Easy to understand and maintain

### 2. **Complete AgentDQN Class**
- All methods are now properly defined inside the AgentDQN class
- No more method binding after class definition
- Cleaner and more Pythonic

### 3. **Comprehensive Unit Tests**
- Tests for all major components
- High test coverage for critical functionality
- Easy to run with `python -m pytest` or the provided test runner

### 4. **Better Documentation**
- Detailed docstrings for all methods and classes
- Type hints for better code clarity
- Clear parameter descriptions

### 5. **Improved Import Structure**
- Proper package structure with `__init__.py`
- Relative imports for better portability
- Clean namespace management

## Usage

### Training a New Agent

```python
import gymnasium as gym
from src import (
    AgentDQN, 
    ReplayBuffer, 
    PreprocessingWrapper,
    map_env_action,
    map_model_action,
    set_config_seed
)

# Set device and seed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_config_seed(42)

# Create environment
env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
env = PreprocessingWrapper(env)

# Create agent
agent = AgentDQN(
    device=device,
    episodes=5000,
    epsilon_min=0.1,
    epsilon_decay=0.9995,
    gamma=0.99,
    update_interval=65,
    checkpoint_min_episodes=500,
    action_space_n=3,
    map_env_action=map_env_action,
    map_model_action=map_model_action
)

# Create replay buffer
replay_buffer = ReplayBuffer(
    size=50000,
    state_shape=torch.Size([4, 84, 84]),
    batch_size=128,
    device=device,
    map_env2model=map_env2model
)

# Train agent
agent.train(
    env=env,
    log_interval=500,
    epsilon=0.995,
    replay_buffer=replay_buffer,
    loss_func=nn.MSELoss()
)
```

### Running Tests

Run all tests:
```bash
cd src
python -m tests.run_tests
```

Run specific test module:
```bash
python -m tests.run_tests network
python -m tests.run_tests agent
python -m tests.run_tests replay
python -m tests.run_tests preprocessing
python -m tests.run_tests utils
```

Or use pytest:
```bash
pytest tests/
```

### Using the Training Script

Run the complete training pipeline:

```bash
python train_pong.py
```

This will:
1. Create the environment with preprocessing
2. Initialize the DQN agent
3. Fill the replay buffer with random experiences
4. Train the agent for the specified number of episodes
5. Save checkpoints during training
6. Evaluate the trained agent

## Module Descriptions

### agent.py
Contains the main `AgentDQN` class with all DQN functionality:
- `__init__`: Initialize agent with networks and optimizer
- `select_action`: Epsilon-greedy action selection
- `train`: Main training loop
- `_train_q_online`: Q-network training step
- `_update_q_target`: Target network update
- `_reduce_epsilon`: Epsilon decay
- `save_training`/`from_checkpoint`: Checkpoint management
- `run`: Agent evaluation
- `fill`: Replay buffer initialization

### network.py
Contains the `Network` class implementing the Deep Q-Network architecture:
- 3 convolutional layers (32, 64, 64 filters)
- 2 fully connected layers (512, num_actions)
- ReLU activations
- Based on the Nature DQN paper

### replay_buffer.py
Contains experience replay implementations:
- `Experience`: Named tuple for experience storage
- `ReplayBuffer`: Standard experience replay
- `BinaryReplayBuffer`: Balanced replay separating wins and losses

### preprocessing.py
Contains `PreprocessingWrapper` for Atari frame preprocessing:
- Frame skipping
- Grayscale conversion
- Resizing (84x84)
- Frame stacking (4 frames)

### utils.py
Utility functions:
- Action mapping functions for Pong
- Random seed management
- Timing decorator
- Environment creation helper
- Visualization helpers

## Configuration

Key hyperparameters and their meanings:

- **episodes**: Number of training episodes
- **epsilon**: Initial exploration rate
- **epsilon_min**: Minimum exploration rate
- **epsilon_decay**: Decay factor for exploration
- **gamma**: Discount factor for future rewards
- **learning_rate**: Learning rate for RMSprop optimizer
- **update_interval**: Episodes between target network updates
- **batch_size**: Number of experiences per training batch
- **buffer_size**: Maximum replay buffer capacity

## Requirements

```
torch>=1.12.0
gymnasium[atari]>=0.28.0
ale-py>=0.8.0
numpy>=1.21.0
matplotlib>=3.5.0
tqdm>=4.64.0
```

## Original Code

This refactored implementation is based on the original code that used method binding after class definition. The refactoring maintains all original functionality while improving:

- Code organization
- Testability
- Documentation
- Maintainability

## License

Same as the original code - this is a refactored version for educational purposes.
