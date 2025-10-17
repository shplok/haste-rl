import pyautogui
import time

print("finding both button positions")
print("\nstep 1: find restart button")
print("  - press ESC in game")
print("  - hover over RESTART button")
print("  - press ENTER when ready\n")

input("press enter when hovering over ABANDON SHARD button...")
time.sleep(2)
abandon_x, abandon_y = pyautogui.position()
print(f"restart button: x={abandon_x}, y={abandon_y}")

input("press enter when hovering over RESTART button...")
time.sleep(2)
restart_x, restart_y = pyautogui.position()
print(f"restart button: x={restart_x}, y={restart_y}")

print("\nstep 2: find new seed button")
print("  - click restart (menu should change)")
print("  - hover over NEW SEED button")
print("  - press ENTER when ready\n")

input("press enter when hovering over NEW SEED button...")
time.sleep(2)
new_seed_x, new_seed_y = pyautogui.position()
print(f"new seed button: x={new_seed_x}, y={new_seed_y}")

print("\n" + "="*50)
print("summary:")
print("="*50)
print(f"restart_button_pos = ({restart_x}, {restart_y})")
print(f"new_seed_button_pos = ({new_seed_x}, {new_seed_y})")
print("\nadd these to haste_env.py")