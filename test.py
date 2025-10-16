from stable_baselines3 import PPO
from haste_env import HasteEnv

# Load environment and model
env = HasteEnv()
model = PPO.load("models/haste_ppo")

# Test
obs, info = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

env.close()