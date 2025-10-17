from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from haste_env import HasteEnv
import time

def make_env(window_index):
    """Create environment factory for each window."""
    def _init():
        return HasteEnv(mouse_sensitivity=500, window_index=window_index)
    return _init

num_envs = 2
print(f"setting up {num_envs} parallel game instances")

env = SubprocVecEnv([make_env(i) for i in range(num_envs)])

print("creating ppo agent")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    tensorboard_log="./logs/",
    device='cpu'
)

checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./models/",
    name_prefix="haste_ppo_parallel",
    save_replay_buffer=False,
    save_vecnormalize=False,
)

print(f"\nstarting training in 10 seconds")
print(f"make sure {num_envs} haste windows are open side-by-side")
print("window 0: left side, window 1: right side")
time.sleep(10)

print("training started")
print(f"{num_envs}x parallel training enabled")

try:
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        progress_bar=True
    )
    
    model.save("models/haste_ppo_parallel_final")

except KeyboardInterrupt:
    print("training interrupted")
    model.save("models/haste_ppo_parallel_interrupted")
finally:
    env.close()

print("training complete")