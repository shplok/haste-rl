# test_env.py
from haste_env import HasteEnv
import time
import cv2

print("="*50)
print("Testing Haste Environment (Simplified)")
print("="*50)

env = HasteEnv()
print("✓ Environment created")

obs, info = env.reset()
print(f"✓ Environment reset")
print(f"  Observation shape: {obs.shape}")

cv2.imwrite("initial_obs.png", obs)

print("\n" + "="*50)
print("Testing Actions")
print("="*50)

test_actions = [
    ([1, 2, 2], "Move forward"),
    ([3, 2, 2], "Strafe left"),
    ([4, 2, 2], "Strafe right"),
    ([1, 4, 2], "Forward + look right"),
    ([1, 0, 2], "Forward + look left"),
    ([1, 2, 0], "Forward + look up"),
    ([1, 2, 4], "Forward + look down"),
    ([0, 2, 2], "Stop"),
]

for i, (action, description) in enumerate(test_actions):
    print(f"\n{i+1}. {description}")
    print(f"   Action: {action}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    cv2.imwrite(f"obs_step_{i+1}.png", obs)
    time.sleep(1.5)

env.close()
print("\n✓ Test complete!")