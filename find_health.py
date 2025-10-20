import pyautogui
import time

print("hover over TOP-LEFT corner of health bar")
time.sleep(3)
x1, y1 = pyautogui.position()

print("hover over BOTTOM-RIGHT corner of health bar")
time.sleep(3)
x2, y2 = pyautogui.position()

print(f"\nHealth bar region:")
print(f"  top: {y1}")
print(f"  left: {x1}")
print(f"  width: {x2 - x1}")
print(f"  height: {y2 - y1}")