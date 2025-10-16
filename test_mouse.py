# test_mouse.py
from pynput.mouse import Controller
import time

mouse = Controller()

print("Testing mouse movement...")
print(f"Current position: {mouse.position}")

sensitivities = [100, 300, 500, 1000, 2000]

for sens in sensitivities:
    print(f"\nTesting sensitivity: {sens}")
    print("Moving right in 3 seconds...")
    time.sleep(3)
    
    start_pos = mouse.position
    mouse.move(sens, 0)
    time.sleep(0.5)
    end_pos = mouse.position
    
    actual_movement = end_pos[0] - start_pos[0]
    print(f"  Commanded: {sens}px, Actual: {actual_movement}px")
    
    # Move back
    mouse.move(-sens, 0)
    time.sleep(1)

print("\nTest complete!")
print("Watch how much the cursor moves for each sensitivity level")
