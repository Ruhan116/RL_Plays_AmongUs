# Multi-Agent Grid Navigation with PPO

A minimal yet complete implementation of multi-agent reinforcement learning where two agents learn to navigate to their respective goal cells on a 5×5 grid using Proximal Policy Optimization (PPO).

## 🎯 Project Overview

This project demonstrates:
- **2 agents** navigating independently on a **5×5 grid**
- Each agent has its **own goal cell** (different locations)
- **PPO algorithm** from Stable-Baselines3 for training
- **PettingZoo** for multi-agent environment framework
- **Pygame** visualization for real-time rendering

### Environment Details

- **Grid Size**: 5×5
- **Agents**: 2 (player_0, player_1)
- **Initial Positions**: 
  - Player 0: Bottom-left (4, 0)
  - Player 1: Top-right (0, 4)
- **Goal Positions**:
  - Player 0: Top-left (0, 0) - Green
  - Player 1: Bottom-right (4, 4) - Red
- **Actions**: 5 discrete actions
  - 0: Move up
  - 1: Move down
  - 2: Move left
  - 3: Move right
  - 4: Stay still
- **Rewards**:
  - +1 for reaching own goal
  - 0 otherwise
- **Episode End**: When both agents reach their goals

## 📁 Project Structure

```
multi_agent_grid/
├── environment/
│   ├── __init__.py
│   └── simple_grid.py       # PettingZoo environment implementation
├── scripts/
│   ├── train.py             # Training script
│   └── evaluate.py          # Evaluation with visualization
├── models/                  # Saved models directory
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Installation

1. **Create a virtual environment** (recommended):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

2. **Install dependencies**:
```powershell
pip install -r requirements.txt
```

### Training

Train the PPO agents for 50,000 timesteps:

```powershell
python scripts/train.py
```

Training output includes:
- Environment setup confirmation
- PPO hyperparameters
- Training progress with episode statistics
- Model save confirmation

**Training time**: ~5-10 minutes on a standard CPU

### Evaluation

Run the trained agents with Pygame visualization:

```powershell
python scripts/evaluate.py
```

The visualization shows:
- **Green square**: Goal for Player 0
- **Red square**: Goal for Player 1
- **Blue circle (A0)**: Player 0
- **Yellow circle (A1)**: Player 1

The agents will autonomously navigate to their respective goals!

## 🎮 Visualization

The Pygame window displays:
- Grid with cells outlined
- Goal cells highlighted with colors
- Agents as colored circles with labels (A0, A1)
- Goals labeled (G0, G1)

Each episode runs until both agents reach their goals, then automatically starts the next episode after a 2-second pause.

## 🧠 How It Works

### 1. Environment (`simple_grid.py`)
- Implements PettingZoo's `AECEnv` (Agent Environment Cycle)
- Turn-based execution where agents act sequentially
- Simple observation: 5×5 grid with agent's position marked
- Actions constrained to grid boundaries

### 2. Training (`train.py`)
- Converts PettingZoo environment to Stable-Baselines3 compatible format using SuperSuit
- Uses PPO algorithm with default hyperparameters
- Agents learn optimal navigation policies through trial and error

### 3. Evaluation (`evaluate.py`)
- Loads trained model
- Runs multiple episodes with visualization
- Agents use learned policy (deterministic mode)

## 🔧 Customization

### Modify Training Parameters

Edit `scripts/train.py`:

```python
model = PPO(
    "MlpPolicy", 
    parallel_env, 
    verbose=1,
    learning_rate=3e-4,      # Adjust learning rate
    n_steps=2048,            # Steps per update
    batch_size=64,           # Batch size
    n_epochs=10,             # Epochs per update
    # ... other parameters
)
```

### Change Grid Size or Goals

Edit `environment/simple_grid.py`:

```python
self.grid_size = 7  # Change grid size

self.goals = {
    "player_0": (0, 0),     # Change goal positions
    "player_1": (6, 6)
}

self.pos = {
    "player_0": (6, 0),     # Change starting positions
    "player_1": (0, 6)
}
```

### Add More Agents

Extend the `agents` list and add corresponding goals:

```python
self.agents = ["player_0", "player_1", "player_2"]
self.goals = {
    "player_0": (0, 0),
    "player_1": (4, 4),
    "player_2": (2, 2)  # New agent
}
```

## 📊 Expected Results

After training, agents should:
- ✅ Navigate directly to their goal cells
- ✅ Avoid unnecessary movements
- ✅ Complete episodes in ~8-10 steps (optimal path is 4-8 steps)
- ✅ Achieve +1 reward consistently

## 🛠️ Troubleshooting

**Issue**: Model not found during evaluation
- **Solution**: Run `python scripts/train.py` first to train the model

**Issue**: Pygame window not responding
- **Solution**: Close the window manually; the script handles cleanup

**Issue**: Import errors
- **Solution**: Ensure all dependencies are installed via `pip install -r requirements.txt`

## 📚 Technologies Used

- **[Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3)**: PPO implementation
- **[PettingZoo](https://github.com/Farama-Foundation/PettingZoo)**: Multi-agent environment framework
- **[SuperSuit](https://github.com/Farama-Foundation/SuperSuit)**: Environment wrappers
- **[Gymnasium](https://github.com/Farama-Foundation/Gymnasium)**: Environment spaces
- **[Pygame](https://www.pygame.org/)**: Visualization
- **[PyTorch](https://pytorch.org/)**: Deep learning backend

## 📝 License

This project is open-source and available for educational purposes.

## 🎓 Learning Resources

- [Stable-Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [PettingZoo Documentation](https://pettingzoo.farama.org/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

---

**Happy Learning! 🚀**
