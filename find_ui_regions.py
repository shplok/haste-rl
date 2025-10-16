# test_ocr_regions.py
from mss import mss
import cv2
import numpy as np
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

monitor = {
    "top": 189,
    "left": 320,
    "width": 1913,
    "height": 1075
}

# Speed region (top half of first grid square)
speed_region = {
    "top": monitor["top"] + 0,
    "left": monitor["left"] + 0,
    "width": 100,
    "height": 50
}

# Rank region (second grid square)
rank_region = {
    "top": monitor["top"] + 100,
    "left": monitor["left"] + 0,
    "width": 100,
    "height": 100
}

print("Capturing in 3 seconds - make sure Haste is visible!")
time.sleep(3)

with mss() as sct:
    # Capture speed region
    speed_img = sct.grab(speed_region)
    speed_img = np.array(speed_img)
    speed_img = cv2.cvtColor(speed_img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite("captured_speed.png", speed_img)
    
    # Capture rank region
    rank_img = sct.grab(rank_region)
    rank_img = np.array(rank_img)
    rank_img = cv2.cvtColor(rank_img, cv2.COLOR_BGRA2BGR)
    cv2.imwrite("captured_rank.png", rank_img)
    
    print("\n✓ Captured regions saved:")
    print("  - captured_speed.png")
    print("  - captured_rank.png")
    
    # Try OCR on speed
    speed_gray = cv2.cvtColor(speed_img, cv2.COLOR_BGR2GRAY)
    _, speed_thresh = cv2.threshold(speed_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("speed_processed.png", speed_thresh)
    
    speed_text = pytesseract.image_to_string(speed_thresh, config='--psm 7 -c tessedit_char_whitelist=0123456789')
    print(f"\n  Speed OCR result: '{speed_text.strip()}'")
    
    # Try OCR on rank
    rank_gray = cv2.cvtColor(rank_img, cv2.COLOR_BGR2GRAY)
    _, rank_thresh = cv2.threshold(rank_gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("rank_processed.png", rank_thresh)
    
    rank_text = pytesseract.image_to_string(rank_thresh, config='--psm 10 -c tessedit_char_whitelist=EDCBAS')
    print(f"  Rank OCR result: '{rank_text.strip()}'")
    
    print("\nCheck the saved images to verify they captured the right areas!")