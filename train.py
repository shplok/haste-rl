from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from haste_env import HasteEnv
import time
import os
import glob

print("checking for existing checkpoints")

env = HasteEnv(mouse_sensitivity=500)

# Try to load most recent checkpoint
checkpoints = glob.glob("models/haste_ppo_*.zip")
if checkpoints:
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"found checkpoint: {latest_checkpoint}")
    print("loading and continuing training...")
    model = PPO.load(latest_checkpoint, env=env)
    reset_timesteps = False
else:
    print("no checkpoints found - creating new model")
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
    reset_timesteps = True

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="haste_ppo",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print("\nstarting training in 7 seconds")
time.sleep(7)

print("training started")

try:
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        progress_bar=True,
        reset_num_timesteps=reset_timesteps
    )
    
    model.save("models/haste_ppo_final")

except KeyboardInterrupt:
    print("training interrupted")
    model.save("models/haste_ppo_interrupted")
finally:
    env.close()

print("training complete")