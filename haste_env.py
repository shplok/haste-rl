import gymnasium as gym
import numpy as np
from mss import mss
from PIL import Image
import cv2
from pynput.keyboard import Controller, Key, Listener
import pydirectinput
import time
import os

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
        
        # Rank region
        self.rank_region = {
            "top": self.monitor["top"] + 100,
            "left": self.monitor["left"] + 0,
            "width": 100,
            "height": 100
        }
        
        # Input controllers
        self.keyboard = Controller()
        self.mouse_sensitivity = mouse_sensitivity
        
        # Track control states
        self.keys_pressed = set()
        
        # Reward tracking
        self.previous_rank = 'E'
        self.rank_values = {'E': 0, 'D': 1, 'C': 2, 'B': 3, 'A': 4, 'S': 5}
        
        # Load rank templates
        self.rank_templates = {}
        if os.path.exists('templates'):
            for rank in ['E', 'D', 'C', 'B', 'A', 'S']:
                template_path = f"templates/rank_{rank}.png"
                if os.path.exists(template_path):
                    self.rank_templates[rank] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        # Death detection
        self.steps_since_rank_visible = 0
        self.max_steps_without_rank = 30
        
        # Manual pause control
        self.paused = False
        self.restart_requested = False
        
        # Start keyboard listener for 'P' key
        self.listener = Listener(on_press=self._on_key_press)
        self.listener.start()
        
        # State
        self.current_step = 0
        self.max_steps = 2000
        self.total_episodes = 0
        
    def _on_key_press(self, key):
        """Handle keyboard input for pause/restart."""
        try:
            if key.char == 'p' or key.char == 'P':
                if self.paused:
                    # Resume
                    self.paused = False
                    self.restart_requested = True
                    print("\n  ▶ RESUMED - Starting new episode")
                else:
                    # Pause
                    self.paused = True
                    print("\n  ⏸ PAUSED - Press 'P' again after manually restarting level")
        except AttributeError:
            pass
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_rank = 'E'
        self.steps_since_rank_visible = 0
        self.total_episodes += 1
        self.restart_requested = False
        
        self._release_all_keys()
        
        print(f"\n{'='*50}")
        print(f"Episode {self.total_episodes} - Ready")
        print('='*50)
        
        time.sleep(1)
        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step."""
        # Check if paused
        while self.paused:
            time.sleep(0.1)
            # When unpaused, trigger a reset
            if self.restart_requested:
                return self._get_observation(), 0, True, False, {'needs_reset': True}
        
        movement = action['movement']
        mouse_x = float(action['mouse_x'][0])
        mouse_y = float(action['mouse_y'][0])
        
        self._take_action(movement, mouse_x, mouse_y)
        time.sleep(0.05)
        
        obs = self._get_observation()
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
    
    def _read_rank(self):
        """Read rank using template matching."""
        if not self.rank_templates:
            return None
        
        try:
            screenshot = self.sct.grab(self.rank_region)
            img = np.array(screenshot)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
            # Extract white letter
            _, mask = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)
            kernel = np.ones((2,2), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            processed = np.zeros_like(img_gray)
            processed[mask == 255] = 255
            
            # Match against templates
            best_match_score = -1
            best_rank = None
            
            for rank, template in self.rank_templates.items():
                result = cv2.matchTemplate(processed, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_match_score:
                    best_match_score = max_val
                    best_rank = rank
            
            if best_match_score > 0.7:
                self.steps_since_rank_visible = 0
                return best_rank
            else:
                self.steps_since_rank_visible += 1
                return None
                
        except Exception as e:
            self.steps_since_rank_visible += 1
            return None
    
    def _calculate_reward(self):
        """Calculate reward based on rank only."""
        current_rank = self._read_rank()
        
        if current_rank is None:
            return -0.01
        
        reward = 0.01
        
        rank_value = self.rank_values.get(current_rank, 0)
        reward += rank_value * 0.2
        
        prev_rank_value = self.rank_values.get(self.previous_rank, 0)
        if rank_value > prev_rank_value:
            reward += 2.0
        elif rank_value < prev_rank_value:
            reward -= 0.5
        
        self.previous_rank = current_rank
        return reward
    
    def _is_game_over(self):
        """Detect if dead or level complete."""
        if self.steps_since_rank_visible >= self.max_steps_without_rank:
            print(f"\n  💀 Game over detected - Press 'P' after manually restarting level")
            self.paused = True
            return True
        return False
    
    def _take_action(self, movement, mouse_x, mouse_y):
        """Execute game controls."""
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
        self.listener.stop()