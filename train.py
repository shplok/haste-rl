from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from haste_env import HasteEnv
import time

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

# Auto-save callback - saves every 10,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="haste_ppo",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print("\nStarting training in 7 seconds...")
print("Make sure Haste is running and you're in a level!")
time.sleep(7)

# Train
print("\n" + "="*50)
print("TRAINING STARTED")
print("="*50)
print("Press 'P' to pause/resume after death")

try:
    model.learn(
        total_timesteps=300,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    # Save final model
    model.save("models/haste_ppo_final")
    print("\n✓ Training complete! Final model saved to models/haste_ppo_final.zip")
    
except KeyboardInterrupt:
    print("\n\n⏹ Training interrupted! Saving model...")
    model.save("models/haste_ppo_interrupted")
    print("✓ Model saved to models/haste_ppo_interrupted.zip")
finally:
    env.close()

print("\nTraining complete!")
print("Check models/ folder for saved checkpoints")