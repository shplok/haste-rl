# test_env.py
from haste_env import HasteEnv
import time
import cv2
import numpy as np

print("="*50)
print("Testing Haste Environment (Continuous Mouse)")
print("="*50)

env = HasteEnv()
print("✓ Environment created")
print(f"  Action space: {env.action_space}")

obs, info = env.reset()
print(f"✓ Environment reset")
print(f"  Observation shape: {obs.shape}")

cv2.imwrite("initial_obs.png", obs)

print("\n" + "="*50)
print("Testing Actions")
print("="*50)

# Test actions with continuous mouse control
test_actions = [
    ({'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}, "Move forward"),
    ({'movement': 3, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}, "Strafe left"),
    ({'movement': 4, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}, "Strafe right"),
    ({'movement': 1, 'mouse_x': np.array([0.5]), 'mouse_y': np.array([0.0])}, "Forward + look right (50%)"),
    ({'movement': 1, 'mouse_x': np.array([1.0]), 'mouse_y': np.array([0.0])}, "Forward + look right (100%)"),
    ({'movement': 1, 'mouse_x': np.array([-0.8]), 'mouse_y': np.array([0.0])}, "Forward + look left (80%)"),
    ({'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([-0.5])}, "Forward + look up (50%)"),
    ({'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.5])}, "Forward + look down (50%)"),
    ({'movement': 0, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}, "Stop"),
]

for i, (action, description) in enumerate(test_actions):
    print(f"\n{i+1}. {description}")
    print(f"   Mouse X: {action['mouse_x'][0]:.2f}, Mouse Y: {action['mouse_y'][0]:.2f}")
    
    obs, reward, terminated, truncated, info = env.step(action)
    
    cv2.imwrite(f"obs_step_{i+1}.png", obs)
    time.sleep(1.0)

env.close()
print("\n✓ Test complete!")
print("\nThe mouse should now move smoothly and continuously.")
print("If it's still too slow/fast, adjust mouse_sensitivity in haste_env.py")