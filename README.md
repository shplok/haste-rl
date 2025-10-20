# haste-rl 

Custom Reinforcement Learning agent for [HASTE](https://store.steampowered.com/app/1796470/Haste/) - Training an AI to master high-speed parkour!

## Project Overview

This project uses **PPO (Proximal Policy Optimization)** to train an AI agent to play HASTE, a fast-paced surfing game. The agent learns through:
- **Visual observations**: 128x128 grayscale screenshots
- **Complex actions**: Movement (WASD), mouse aiming, fast-fall (space), and speed boost (left-click)
- **Reward system**: Rank improvements, damage avoidance, and survival time

## Features

- **Computer Vision Integration**: Template matching for rank detection, lives tracking, and health monitoring
- **Automated Level Restarting**: Agent automatically restarts levels with new seeds for diverse training
- **Death Detection**: Multi-layered game-over detection (death screen, lives, health)
- **Dynamic Reward Shaping**: Rewards for rank improvements, penalties for damage and deaths
- **Checkpoint System**: Auto-saves every 10k steps with resume capability

## Tech Stack

- **Gymnasium**: Custom environment framework
- **Stable-Baselines3**: PPO implementation
- **OpenCV**: Computer vision for game state detection
- **MSS**: Fast screen capture
- **pynput/pydirectinput**: Game control automation

## Requirements
```bash
pip install gymnasium
pip install stable-baselines3
pip install opencv-python
pip install mss
pip install pynput
pip install pydirectinput
pip install pyautogui
pip install numpy
pip install pillow
```

## Setup

### 1. Game Configuration
- Launch HASTE in **windowed mode** at **1920x1080** resolution
- Do not move window, all the template coords are set for 1080p
- Start a level before training begins

## Training

### Start Training (Auto-resumes from checkpoints)
```bash
python train.py
```

The script will:
1. Check for existing checkpoints in `models/`
2. Load the most recent checkpoint (if found)
3. Continue training from that point
4. Auto-save every 10,000 steps

### Training from Scratch
Delete all files in `models/` folder to start fresh.

### Monitor Progress
```bash
tensorboard --logdir=./logs/
```

View training metrics at `http://localhost:6006`

## Environment Details

### Observation Space
- **Type**: Grayscale image
- **Size**: 128x128 pixels
- **Range**: 0-255

### Action Space (Continuous)
5 values in range:
- `action[0]`: Movement (0-4) → 0=None, 1=W, 2=S, 3=A, 4=D
- `action[1]`: Mouse X (-1.0 to 1.0)
- `action[2]`: Mouse Y (-1.0 to 1.0)
- `action[3]`: Fast Fall (0-1) → Space bar
- `action[4]`: Speed Boost (0-1) → Left click

### Reward Structure

**Positive Rewards:**
- `+0.01`: Per step with visible rank
- `+0.2 * rank_value`: Higher ranks (E=0, D=0.2, C=0.4, B=0.8, A=1.0, S=1.2)
- `+2.0`: Rank improvement bonus

**Negative Penalties:**
- `-0.01`: Per step without visible rank
- `-0.5`: Rank decrease
- `-5.0`: Per health quarter lost
- `-30.0 - (episode_reward * 0.5)`: Death or losing life

### Episode Termination
Episodes end when:
- Agent reaches 1 life remaining
- Death screen detected
- Rank not visible for 150 steps
- 2000 steps elapsed (truncation)

## Project Structure
```
haste-rl/
├── haste_env.py              # Custom Gymnasium environment
├── train.py                  # Training script with auto-resume
├── capture_rank_templates.py # UI element capture tool
├── capture_lives_templates.py
├── find_positions.py         # Screen coordinate finder
├── test_health_detection.py  # Health bar testing
├── models/                   # Saved model checkpoints
├── logs/                     # TensorBoard training logs
└── templates/                # Captured UI templates
    ├── rank_E.png
    ├── rank_D.png
    ├── ...
    ├── lives_1.png
    └── ...
```

## Customization

### Adjust Reward Weights
Edit `haste_env.py`:
```python
self.rank_values = {'E': 0, 'D': 1, 'C': 2, 'B': 4, 'A': 5, 'S': 6}
damage_penalty = -5.0 * quarters_lost  # Health damage
death_penalty = -30.0 - (self.episode_reward * 0.5)  # Death
```

### Change Hyperparameters
Edit `train.py`:
```python
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,    # Learning rate
    n_steps=2048,          # Steps per update
    batch_size=64,         # Batch size
    n_epochs=10,           # Optimization epochs
    gamma=0.99,            # Discount factor
    ...
)
```

### Mouse Sensitivity
```python
env = HasteEnv(mouse_sensitivity=500)  # Adjust for your preference
```

## Training Tips

- **Early training (0-50k steps)**: Agent will flail randomly
- **Mid training (50k-200k steps)**: Should start moving forward consistently
- **Advanced training (200k-500k+)**: May achieve C/B ranks
- **Expected time**: ~3-5 FPS → 100k steps takes ~9-12 hours

## Troubleshooting

### Agent not controlling game
- Ensure game window is in focus and active
- Check screen coordinates match your setup
- Verify game is in windowed mode

### Template matching fails
- Recapture templates at your resolution
- Adjust threshold values in `_read_rank()` and `_read_lives()`
- Check `rank_region` and `lives_region` coordinates

### Training too slow
- Close unnecessary applications
- Lower game graphics settings
- Reduce `time.sleep()` values (may affect stability)

### Auto-restart not working
- Verify button coordinates with `find_positions.py`
- Adjust sleep timings in `_auto_restart()` for slower systems
- Check that ESC opens the correct menu

## Learning Resources

- [Stable-Baselines3 Docs](https://stable-baselines3.readthedocs.io/)
- [PPO Algorithm](https://arxiv.org/abs/1707.06347)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## License

This project is for educational purposes. HASTE is property of its respective owners. Please don't use this model to gain a competitive advantage or farm achievements! This is just for fun.

## Contributing

Feel free to open issues or submit pull requests for improvements!

---

**Check out HASTE on Steam!**  
https://store.steampowered.com/app/1796470/Haste/
