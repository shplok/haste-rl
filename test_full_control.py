from haste_env import HasteEnv
import numpy as np
import time

print("="*50)
print("Full Control Test")
print("="*50)
print("\nMake sure:")
print("  1. Haste is running and in a level")
print("  2. Click inside the game window")
print("  3. Don't touch mouse/keyboard during test")
print("\nStarting in 5 seconds...")
time.sleep(5)

env = HasteEnv(mouse_sensitivity=500)
obs, info = env.reset()

print("\n✓ Environment ready!")
print(f"  Observation shape: {obs.shape}")

# Test sequence
tests = [
    ("Moving forward", 10, {'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}),
    ("Turning right", 15, {'movement': 1, 'mouse_x': np.array([0.3]), 'mouse_y': np.array([0.0])}),
    ("Turning left", 15, {'movement': 1, 'mouse_x': np.array([-0.3]), 'mouse_y': np.array([0.0])}),
    ("Strafing right", 10, {'movement': 4, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}),
    ("Strafing left", 10, {'movement': 3, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}),
    ("Looking up", 8, {'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([-0.3])}),
    ("Looking down", 8, {'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.3])}),
]

for description, steps, action in tests:
    print(f"\n{description}...")
    for _ in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        time.sleep(0.1)
    time.sleep(0.5)

print("\n✓ All tests complete!")
print("\nDid the character move correctly?")

env.close()