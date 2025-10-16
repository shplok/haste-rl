from haste_env import HasteEnv
import numpy as np
import time

print("Testing reward detection...")
print("Starting in 5 seconds - make sure game is visible!")
time.sleep(5)

env = HasteEnv(mouse_sensitivity=500)
obs, info = env.reset()

print("\nRunning for 20 steps and printing speed/rank/reward...")

for i in range(20):
    # Move forward
    action = {'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}
    obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"Step {i+1}: Speed={env.previous_speed}, Rank={env.previous_rank}, Reward={reward:.3f}")
    
    time.sleep(0.2)

env.close()
print("\nIf speed/rank show 0/E the whole time, adjust the OCR regions!")