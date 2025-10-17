# capture_rank_templates.py
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

rank_region = {
    "top": monitor["top"] + 100,
    "left": monitor["left"] + 0,
    "width": 100,
    "height": 100
}

print("="*50)
print("Rank Template Capture (White Letter Only)")
print("="*50)
print("\nWe'll extract ONLY the white letter, ignoring background.\n")

ranks = ['E', 'D', 'C', 'B', 'A', 'S']

with mss() as sct:
    for rank in ranks:
        input(f"Get rank {rank} showing, then press Enter...")
        time.sleep(1)  # Wait a moment to ensure stable capture
        # Capture
        screenshot = sct.grab(rank_region)
        img = np.array(screenshot)
        img_color = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # Threshold to extract ONLY bright white pixels (the letter)
        # White text is typically 200-255 in grayscale
        _, mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to clean up
        kernel = np.ones((2,2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
        
        # Create white letter on black background
        template = np.zeros_like(img_gray)
        template[mask == 255] = 255
        
        # Save template
        cv2.imwrite(f"templates/rank_{rank}.png", template)
        
        # Save comparison for verification
        comparison = np.hstack([img_gray, mask, template])
        cv2.imwrite(f"templates/rank_{rank}_process.png", comparison)
        
        print(f"  ✓ Saved rank_{rank}.png")
        print(f"    Check rank_{rank}_process.png to verify extraction")

print("\n✓ All templates saved!")
print("\nOpen templates/rank_*_process.png files to verify:")
print("  - Left: Original")
print("  - Middle: Threshold mask")
print("  - Right: Final template (should show ONLY the white letter)")
