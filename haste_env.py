import gymnasium as gym
import numpy as np
from mss import mss
from PIL import Image
import cv2
from pynput.keyboard import Controller, Key
import time

class HasteEnv(gym.Env):
    """Custom Environment for Haste game."""
    
    def __init__(self):
        super(HasteEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 0=nothing, 1=forward(W), 2=left(A), 3=right(D), 4=jump(Space)
        self.action_space = gym.spaces.Discrete(5)
        
        # Observations: 84x84 grayscale image
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
        
        # Input controller
        self.keyboard = Controller()
        
        # State
        self.current_step = 0
        self.max_steps = 1000
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        
        # TODO: Add logic to restart the game
        time.sleep(1)  # Wait for game to reset
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step."""
        # Take action
        self._take_action(action)
        
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
        
        # Resize to 84x84
        img = cv2.resize(img, (84, 84))
        
        return img
    
    def _take_action(self, action):
        """Execute keyboard action."""
        # Release all keys first
        # (In real implementation, track which keys are pressed)
        
        if action == 1:  # Forward
            self.keyboard.press('w')
            time.sleep(0.1)
            self.keyboard.release('w')
        elif action == 2:  # Left
            self.keyboard.press('a')
            time.sleep(0.1)
            self.keyboard.release('a')
        elif action == 3:  # Right
            self.keyboard.press('d')
            time.sleep(0.1)
            self.keyboard.release('d')
        elif action == 4:  # Jump
            self.keyboard.press(Key.space)
            time.sleep(0.1)
            self.keyboard.release(Key.space)
        # action == 0: do nothing
    
    def close(self):
        """Cleanup."""
        pass

# Quick test
if __name__ == "__main__":
    env = HasteEnv()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action}, Reward: {reward}")
        
        if terminated or truncated:
            break
    
    env.close()