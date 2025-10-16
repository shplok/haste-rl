import gymnasium as gym
import numpy as np
from mss import mss
from PIL import Image
import cv2
from pynput.keyboard import Controller, Key
import pydirectinput
import time

class HasteEnv(gym.Env):
    """Custom Environment for Haste game."""
    
    def __init__(self, mouse_sensitivity=2000):
        super(HasteEnv, self).__init__()
        
        # Define action space with CONTINUOUS mouse control
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

        # Mouse sensitivity
        self.mouse_sensitivity = mouse_sensitivity
        
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
        reward = 1.0
        
        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """Capture and process screen."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = cv2.resize(img, (128, 128))
        return img
    
    def _take_action(self, movement, mouse_x, mouse_y):
        """Execute game controls."""
        
        # 1. Handle movement (WASD)
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
        
        # 2. Handle mouse with PyAutoGUI's moveRel (relative movement)
        # This works better with games that capture the cursor
        mouse_dx = int(mouse_x * self.mouse_sensitivity)
        mouse_dy = int(mouse_y * self.mouse_sensitivity)
        
        if mouse_dx != 0 or mouse_dy != 0:
            # Use relative movement with duration for smoothness
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

# Quick test
if __name__ == "__main__":
    print("Starting test in 5 seconds - click in Haste window!")
    time.sleep(5)
    
    env = HasteEnv(mouse_sensitivity=1000)
    obs, info = env.reset()
    
    print("Looking right...")
    for _ in range(10):
        action = {'movement': 1, 'mouse_x': np.array([1.0]), 'mouse_y': np.array([0.0])}
        obs, reward, terminated, truncated, info = env.step(action)
    
    time.sleep(1)
    
    print("Looking left...")
    for _ in range(10):
        action = {'movement': 1, 'mouse_x': np.array([-1.0]), 'mouse_y': np.array([0.0])}
        obs, reward, terminated, truncated, info = env.step(action)
    
    env.close()
    print("Test complete!")