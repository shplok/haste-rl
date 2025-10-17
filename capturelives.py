# capture_hearts.py
from mss import mss
import cv2
import numpy as np
import time
import os

os.makedirs('templates', exist_ok=True)

monitor = {
    "top": 189,
    "left": 320,
    "width": 1913,
    "height": 1075
}

# Lives at bottom center
lives_region = {
    "top": monitor["top"] + monitor["height"] - 245,  
    "left": monitor["left"] + (monitor["width"] // 2) - 55,  # Center, 400px wide
    "width": 100,
    "height": 20
}


def extract_red(img):
    """Extract red pixels from image."""
    # Convert BGR to HSV (better for color detection)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Red color range in HSV
    # Red wraps around in HSV, so we need two ranges
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])
    
    # Create masks for red
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = mask1 + mask2
    
    # Create output image (white hearts on black background)
    result = np.zeros_like(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    result[red_mask > 0] = 255
    
    return result

print("finding hearts region")
print("starting in 3 seconds...")
time.sleep(3)

with mss() as sct:
    screenshot = sct.grab(lives_region)
    img = np.array(screenshot)
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    # Save original
    cv2.imwrite("templates/lives_region_original.png", img)
    
    # Save with red extraction
    red_only = extract_red(img)
    cv2.imwrite("templates/lives_region_red.png", red_only)
    
    print("saved:")
    print("  lives_region_original.png (full color)")
    print("  lives_region_red.png (red hearts only)")
    print("\ncheck if red hearts are visible in lives_region_red.png")

input("\npress enter to capture life templates...")

life_counts = [4, 3, 2, 1]

with mss() as sct:
    for lives in life_counts:
        input(f"get to {lives} lives and press enter...")
        time.sleep(2)  # Give a moment to switch
        screenshot = sct.grab(lives_region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        
        # Save original
        cv2.imwrite(f"templates/lives_{lives}_original.png", img)
        
        # Save red extraction (this is what we'll use for matching)
        red_only = extract_red(img)
        cv2.imwrite(f"templates/lives_{lives}.png", red_only)
        
        print(f"  saved lives_{lives}.png (red hearts only)")

print("\ndone! check templates/ folder")
print("the lives_X.png files should show white hearts on black background")