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
        
        # Define action space with CONTINUOUS mouse control
        # This is a Dict space with:
        # - movement: discrete (0-4)
        # - mouse_x: continuous (-1.0 to 1.0)
        # - mouse_y: continuous (-1.0 to 1.0)
        
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
        
        # Input controllers
        self.keyboard = Controller()
        self.mouse = MouseController()
        
        # Mouse sensitivity (adjust if needed)
        self.mouse_sensitivity = 300  # Pixels per action
        
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
        # Parse action
        movement = action['movement']
        mouse_x = float(action['mouse_x'][0])
        mouse_y = float(action['mouse_y'][0])
        
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
        
        # 2. Handle CONTINUOUS mouse movement
        # mouse_x and mouse_y are now in range [-1.0, 1.0]
        # Convert to pixel movement
        mouse_dx = int(mouse_x * self.mouse_sensitivity)
        mouse_dy = int(mouse_y * self.mouse_sensitivity)
        
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
    
    # Test some actions with continuous mouse
    print("\nTesting controls...")
    
    # Move forward
    print("Forward")
    action = {'movement': 1, 'mouse_x': np.array([0.0]), 'mouse_y': np.array([0.0])}
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(1)
    
    # Look right (continuous)
    print("Look right (50%)")
    action = {'movement': 1, 'mouse_x': np.array([0.5]), 'mouse_y': np.array([0.0])}
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.5)
    
    # Look right more
    print("Look right (100%)")
    action = {'movement': 1, 'mouse_x': np.array([1.0]), 'mouse_y': np.array([0.0])}
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.5)
    
    # Look left
    print("Look left")
    action = {'movement': 1, 'mouse_x': np.array([-0.8]), 'mouse_y': np.array([0.0])}
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.5)
    
    env.close()
    print("\nTest complete!")