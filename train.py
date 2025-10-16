from stable_baselines3 import PPO
from haste_env import HasteEnv

# Create environment
env = HasteEnv()

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

# Train
print("Starting training...")
model.learn(total_timesteps=10000)

# Save
model.save("models/haste_ppo")
print("Model saved!")