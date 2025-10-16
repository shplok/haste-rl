# test_env.py
from haste_env import HasteEnv
import time
import cv2

print("="*50)
print("Testing Haste Environment (No Sprint)")
print("="*50)

env = HasteEnv()
print("✓ Environment created")

obs, info = env.reset()
print(f"✓ Environment reset")
print(f"  Observation shape: {obs.shape}")

cv2.imwrite("initial_obs.png", obs)
print("✓ Saved initial observation")

print("\n" + "="*50)
print("Testing Actions (Make sure Haste is in focus!)")
print("="*50)

# Updated test actions (no sprint!)
test_actions = [
    ([1, 0, 2, 2], "Move forward (auto-sprint)"),
    ([3, 0, 2, 2], "Strafe left"),
    ([4, 0, 2, 2], "Strafe right"),
    ([1, 1, 2, 2], "Jump while moving forward"),
    ([1, 0, 4, 2], "Move forward + look right"),
    ([1, 0, 0, 2], "Move forward + look left"),
    ([0, 0, 2, 2], "Stop (release all)"),
]

for i, (action, description) in enumerate(test_actions):
    print(f"\n{i+1}. {description}")
    print(f"   Action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"   Reward: {reward}")
    
    cv2.imwrite(f"obs_step_{i+1}.png", obs)
    time.sleep(1.5)
    
    if terminated or truncated:
        break

print("\n" + "="*50)
print("Test Complete!")
print("="*50)

env.close()