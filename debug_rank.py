# debug_rank.py
from mss import mss
import cv2
import numpy as np
import pytesseract
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

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

print("Play the game for a bit to get different ranks...")
print("Capturing rank every 2 seconds for 20 seconds...")

with mss() as sct:
    for i in range(10):
        time.sleep(2)
        
        # Capture
        rank_img = sct.grab(rank_region)
        rank_img = np.array(rank_img)
        rank_gray = cv2.cvtColor(rank_img, cv2.COLOR_BGRA2GRAY)
        
        # Save original
        cv2.imwrite(f"rank_original_{i}.png", rank_gray)
        
        # Try different preprocessing methods
        
        # Method 1: Simple threshold
        _, thresh1 = cv2.threshold(rank_gray, 127, 255, cv2.THRESH_BINARY)
        cv2.imwrite(f"rank_thresh_{i}.png", thresh1)
        text1 = pytesseract.image_to_string(thresh1, config='--psm 10 -c tessedit_char_whitelist=EDCBAS')
        
        # Method 2: Inverse threshold (white text on dark bg)
        _, thresh2 = cv2.threshold(rank_gray, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.imwrite(f"rank_thresh_inv_{i}.png", thresh2)
        text2 = pytesseract.image_to_string(thresh2, config='--psm 10 -c tessedit_char_whitelist=EDCBAS')
        
        # Method 3: Adaptive threshold
        thresh3 = cv2.adaptiveThreshold(rank_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imwrite(f"rank_adaptive_{i}.png", thresh3)
        text3 = pytesseract.image_to_string(thresh3, config='--psm 10 -c tessedit_char_whitelist=EDCBAS')
        
        # Method 4: Try different PSM modes
        text4 = pytesseract.image_to_string(thresh1, config='--psm 8 -c tessedit_char_whitelist=EDCBAS')
        text5 = pytesseract.image_to_string(thresh1, config='--psm 7 -c tessedit_char_whitelist=EDCBAS')
        
        print(f"\nCapture {i+1}:")
        print(f"  Method 1 (BINARY): '{text1.strip()}'")
        print(f"  Method 2 (BINARY_INV): '{text2.strip()}'")
        print(f"  Method 3 (ADAPTIVE): '{text3.strip()}'")
        print(f"  Method 4 (PSM 8): '{text4.strip()}'")
        print(f"  Method 5 (PSM 7): '{text5.strip()}'")

print("\nDone! Check the saved images to see which preprocessing works best.")
print("Look for files like rank_thresh_inv_X.png, rank_adaptive_X.png, etc.")