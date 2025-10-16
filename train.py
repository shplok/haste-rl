from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from haste_env import HasteEnv
import time

print("="*50)
print("Training Haste RL Agent")
print("="*50)
print("\n Instructions:")
print("  - When agent dies, training will PAUSE")
print("  - Manually restart the level in game")
print("  - Press 'P' to RESUME training")
print("  - Press Ctrl+C to stop and save\n")

# Create environment
print("Creating environment...")
env = HasteEnv(mouse_sensitivity=500)

# Create PPO agent
print("Creating PPO agent...")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    tensorboard_log="./logs/"
)

print("\nStarting training in 10 seconds...")
print("Make sure Haste is running and you're in a level!")
time.sleep(10)

# Train
print("\n" + "="*50)
print("TRAINING STARTED")
print("="*50)
print("Press 'P' to pause/resume after death\n")

try:
    model.learn(total_timesteps=100000, progress_bar=True)
    
    # Save model
    model.save("models/haste_ppo_100k")
    print("\n✓ Training complete! Model saved to models/haste_ppo_100k.zip")
    
except KeyboardInterrupt:
    print("\n\n⏹ Training interrupted! Saving model...")
    model.save("models/haste_ppo_interrupted")
    print("✓ Model saved to models/haste_ppo_interrupted.zip")
finally:
    env.close()

print("\nTraining complete!")