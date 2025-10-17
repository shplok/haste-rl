# capture_lives_720p.py
from mss import mss
import cv2
import numpy as np
import time
import os

os.makedirs('templates_720p', exist_ok=True)

monitor = {
    "top": 0,
    "left": 0,
    "width": 1280,
    "height": 720
}

# YOUR EXACT LIVES REGION
# Top-left: (600, 560), Bottom-right: (672, 580)
lives_region = {
    "top": 595,
    "left": 600,
    "width": 72,  # 672 - 600
    "height": 18 # 580 - 560
}

def extract_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = mask1 + mask2
    
    result = np.zeros(img.shape[:2], dtype=np.uint8)
    result[red_mask > 0] = 255
    return result

print("capturing lives templates at 720p")
print("make sure game window is at top-left corner (0, 0)")

with mss() as sct:
    for lives in [4, 3, 2, 1]:
        input(f"get to {lives} lives and press enter...")
        time.sleep(2)  # brief pause to ensure stable capture
        
        screenshot = sct.grab(lives_region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        red_only = extract_red(img)
        
        cv2.imwrite(f"templates_720p/lives_{lives}.png", red_only)
        print(f"  saved lives_{lives}.png")

print("done")