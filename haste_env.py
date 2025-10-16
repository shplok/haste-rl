import gymnasium as gym
import numpy as np
from mss import mss
from PIL import Image
import cv2
from pynput.keyboard import Controller, Key
import pydirectinput
import pytesseract
import time
import re

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"

class HasteEnv(gym.Env):
    """Custom Environment for Haste game."""
    
    def __init__(self, mouse_sensitivity=500):
        super(HasteEnv, self).__init__()
        
        # Define action space
        self.action_space = gym.spaces.Dict({
            'movement': gym.spaces.Discrete(5),
            'mouse_x': gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            'mouse_y': gym.spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        })
        
        # Observations: 128x128 grayscale image
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128), dtype=np.uint8
        )
        
        # Screen capture setup
        self.sct = mss()
        self.monitor = {
            "top": 189,
            "left": 320,
            "width": 1913,
            "height": 1075
        }
        
        # UI Regions (based on grid - top-left is 0-100, next down is 100-200)
        # Speed: top half of first grid square (0-50 vertical)
        self.speed_region = {
            "top": self.monitor["top"] + 0,      # Top of first grid
            "left": self.monitor["left"] + 0,    # Left edge
            "width": 100,                        # One grid square wide
            "height": 50                         # Top half of grid square
        }
        
        # Rank: second grid square down (100-200 vertical)
        self.rank_region = {
            "top": self.monitor["top"] + 100,    # Second grid square
            "left": self.monitor["left"] + 0,    # Left edge
            "width": 100,                        # One grid square wide
            "height": 100                        # Full grid square height
        }
        
        # Input controllers
        self.keyboard = Controller()
        self.mouse_sensitivity = mouse_sensitivity
        
        # Track control states
        self.keys_pressed = set()
        
        # Reward tracking
        self.previous_speed = 0
        self.previous_rank = 'E'
        self.rank_values = {'E': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4, 'S': 5}
        
        # State
        self.current_step = 0
        self.max_steps = 1000
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_speed = 0
        self.previous_rank = 'E'
        
        # Release all keys
        self._release_all_keys()
        
        time.sleep(1)
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step."""
        movement = action['movement']
        mouse_x = float(action['mouse_x'][0])
        mouse_y = float(action['mouse_y'][0])
        
        self._take_action(movement, mouse_x, mouse_y)
        
        time.sleep(0.05)
        
        obs = self._get_observation()
        
        # Calculate reward based on speed and rank
        reward = self._calculate_reward()
        
        self.current_step += 1
        terminated = self._is_game_over()
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """Capture and process screen."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = cv2.resize(img, (128, 128))
        return img
    
    def _read_speed(self):
        """Read speed value from screen using OCR."""
        try:
            # Capture speed region
            screenshot = self.sct.grab(self.speed_region)
            img = np.array(screenshot)
            
            # Preprocess for better OCR
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
            # Try different thresholds
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            # Save debug image (comment out after testing)
            cv2.imwrite("debug_speed.png", img)
            
            # OCR with config for single line of digits
            text = pytesseract.image_to_string(img, config='--psm 7 -c tessedit_char_whitelist=0123456789')
            
            # Extract number
            numbers = re.findall(r'\d+', text)
            if numbers:
                speed = int(numbers[0])
                return min(speed, 200)  # Cap at 200
            
        except Exception as e:
            print(f"Speed OCR error: {e}")
        
        return self.previous_speed  # Return previous if OCR fails
    
    def _read_rank(self):
        """Read rank from screen using OCR."""
        try:
            # Capture rank region
            screenshot = self.sct.grab(self.rank_region)
            img = np.array(screenshot)
            
            # Preprocess
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            # Save debug image (comment out after testing)
            cv2.imwrite("debug_rank.png", img)
            
            # OCR with config for single character
            text = pytesseract.image_to_string(img, config='--psm 10 -c tessedit_char_whitelist=EDCBAS').strip().upper()
            
            # Extract rank letter
            for char in text:
                if char in self.rank_values:
                    return char
            
        except Exception as e:
            print(f"Rank OCR error: {e}")
        
        return self.previous_rank  # Return previous if OCR fails
    
    def _calculate_reward(self):
        """Calculate reward based on speed and rank."""
        # Read current values
        current_speed = self._read_speed()
        current_rank = self._read_rank()
        
        # Base reward: small positive for survival
        reward = 0.01
        
        # Speed reward (normalized to 0-1, where 200 speed = max)
        speed_reward = current_speed / 200.0
        reward += speed_reward * 0.5  # Speed contributes up to 0.5 reward
        
        # Speed improvement bonus
        if current_speed > self.previous_speed:
            reward += 0.1
        
        # Rank reward
        rank_value = self.rank_values.get(current_rank, 0)
        reward += rank_value * 0.1  # Each rank tier adds 0.1
        
        # Rank improvement bonus
        prev_rank_value = self.rank_values.get(self.previous_rank, 0)
        if rank_value > prev_rank_value:
            reward += 1.0  # Big bonus for improving rank!
        
        # Update previous values
        self.previous_speed = current_speed
        self.previous_rank = current_rank
        
        return reward
    
    def _is_game_over(self):
        """Detect if game is over (TODO: implement detection)."""
        # TODO: Add detection for falling off map, level complete, etc.
        return False
    
    def _take_action(self, movement, mouse_x, mouse_y):
        """Execute game controls."""
        
        # Handle movement (WASD)
        for key in ['w', 'a', 's', 'd']:
            if key in self.keys_pressed:
                self.keyboard.release(key)
                self.keys_pressed.discard(key)
        
        if movement == 1:
            self.keyboard.press('w')
            self.keys_pressed.add('w')
        elif movement == 2:
            self.keyboard.press('s')
            self.keys_pressed.add('s')
        elif movement == 3:
            self.keyboard.press('a')
            self.keys_pressed.add('a')
        elif movement == 4:
            self.keyboard.press('d')
            self.keys_pressed.add('d')
        
        # Handle mouse
        mouse_dx = int(mouse_x * self.mouse_sensitivity)
        mouse_dy = int(mouse_y * self.mouse_sensitivity)
        
        if mouse_dx != 0 or mouse_dy != 0:
            pydirectinput.moveRel(mouse_dx, mouse_dy, relative=True)
    
    def _release_all_keys(self):
        """Release all pressed keys."""
        for key in list(self.keys_pressed):
            if key in ['w', 'a', 's', 'd']:
                self.keyboard.release(key)
        self.keys_pressed.clear()
    
    def close(self):
        """Cleanup."""
        self._release_all_keys()