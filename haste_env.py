import gymnasium as gym
import numpy as np
from mss import mss
from PIL import Image
import cv2
from pynput.keyboard import Controller, Key
from pynput.mouse import Controller as MouseController
import time

class HasteEnv(gym.Env):
    """Custom Environment for Haste game."""
    
    def __init__(self):
        super(HasteEnv, self).__init__()
        
        # Define action space (FURTHER SIMPLIFIED - no jump, better mouse)
        # [movement, mouse_x, mouse_y]
        # movement: 0=none, 1=forward, 2=back, 3=left, 4=right
        # mouse_x: -2 to 2 (left to right, discrete steps)
        # mouse_y: -2 to 2 (up to down, discrete steps)
        
        self.action_space = gym.spaces.MultiDiscrete([5, 5, 5])
        
        # Observations: 128x128 grayscale image
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128), dtype=np.uint8
        )
        
        # Screen capture setup - YOUR DIMENSIONS
        self.sct = mss()
        self.monitor = {
            "top": 189,
            "left": 320,
            "width": 1913,
            "height": 1075
        }
        
        # Input controllers
        self.keyboard = Controller()
        self.mouse = MouseController()
        
        # Track control states
        self.keys_pressed = set()
        
        # State
        self.current_step = 0
        self.max_steps = 1000
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # Release all keys
        self._release_all_keys()
        
        # TODO: Add logic to restart the game
        time.sleep(1)  # Wait for game to reset
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step."""
        # Parse action (no jump!)
        movement, mouse_x, mouse_y = action
        
        # Execute action
        self._take_action(movement, mouse_x, mouse_y)
        
        # Small delay to let game update
        time.sleep(0.05)
        
        # Get observation
        obs = self._get_observation()
        
        # Calculate reward (simple: survived = +1)
        reward = 1.0
        
        # Check if done
        self.current_step += 1
        terminated = False  # TODO: detect game over
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """Capture and process screen."""
        # Capture screen
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        
        # Convert to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        
        # Resize to 128x128
        img = cv2.resize(img, (128, 128))
        
        return img
    
    def _take_action(self, movement, mouse_x, mouse_y):
        """Execute game controls."""
        
        # 1. Handle movement (WASD)
        # First release previous movement keys
        for key in ['w', 'a', 's', 'd']:
            if key in self.keys_pressed:
                self.keyboard.release(key)
                self.keys_pressed.discard(key)
        
        # Press new movement key
        if movement == 1:  # Forward
            self.keyboard.press('w')
            self.keys_pressed.add('w')
        elif movement == 2:  # Back
            self.keyboard.press('s')
            self.keys_pressed.add('s')
        elif movement == 3:  # Left (strafe when auto-sprinting)
            self.keyboard.press('a')
            self.keys_pressed.add('a')
        elif movement == 4:  # Right (strafe when auto-sprinting)
            self.keyboard.press('d')
            self.keys_pressed.add('d')
        # movement == 0: no movement
        
        # 2. Handle mouse movement (IMPROVED)
        # Convert discrete action to pixel movement
        # Increased sensitivity and made it more responsive
        mouse_dx = (mouse_x - 2) * 100  # Doubled from 50 to 100
        mouse_dy = (mouse_y - 2) * 100
        
        if mouse_dx != 0 or mouse_dy != 0:
            self.mouse.move(mouse_dx, mouse_dy)
    
    def _release_all_keys(self):
        """Release all pressed keys."""
        for key in list(self.keys_pressed):
            if key in ['w', 'a', 's', 'd']:
                self.keyboard.release(key)
        
        self.keys_pressed.clear()
    
    def close(self):
        """Cleanup."""
        self._release_all_keys()

# Quick test
if __name__ == "__main__":
    env = HasteEnv()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    # Test some actions
    print("\nTesting controls...")
    
    # Move forward
    print("Forward")
    obs, reward, terminated, truncated, info = env.step([1, 2, 2])  # forward, center mouse
    time.sleep(1)
    
    # Strafe left
    print("Strafe left")
    obs, reward, terminated, truncated, info = env.step([3, 2, 2])  # left, center mouse
    time.sleep(1)
    
    # Look right
    print("Look right")
    obs, reward, terminated, truncated, info = env.step([1, 4, 2])  # forward, look right
    time.sleep(1)
    
    # Look left
    print("Look left")
    obs, reward, terminated, truncated, info = env.step([1, 0, 2])  # forward, look left
    time.sleep(1)
    
    env.close()
    print("\nTest complete!")