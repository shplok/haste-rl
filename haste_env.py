import gymnasium as gym
import numpy as np
from mss import mss
from PIL import Image
import cv2
from pynput.keyboard import Controller, Key
import pydirectinput
import pyautogui
import time
import os

class HasteEnv(gym.Env):
    """Custom Environment for Haste game."""
    
    def __init__(self, mouse_sensitivity=500):
        super(HasteEnv, self).__init__()
        
        # expanded action space
        # Box with 5 values: [movement, mouse_x, mouse_y, fast_fall, speed_boost]
        # movement: 0-4 (will be discretized)
        # mouse_x: -1.0 to 1.0
        # mouse_y: -1.0 to 1.0
        # fast_fall: 0-1 (0=release space, 1=hold space)
        # speed_boost: 0-1 (0=release click, 1=hold left click)
        self.action_space = gym.spaces.Box(
            low=np.array([0, -1.0, -1.0, 0, 0]),
            high=np.array([4, 1.0, 1.0, 1, 1]),
            dtype=np.float32
        )
        
        # Observations: 128x128 grayscale image
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(128, 128, 1), dtype=np.uint8
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
        
        # Lives region (bottom center)
        self.lives_region = {
            "top": self.monitor["top"] + self.monitor["height"] - 245,  
            "left": self.monitor["left"] + (self.monitor["width"] // 2) - 55,
            "width": 100,
            "height": 20
        }

        # Health bar region 
        self.health_region = {
            "top": 998,
            "left": 1169,
            "width": 205,
            "height": 20
        }
        # 4 quarters - will need manually updated when i level up max health
        self.previous_health_quarters = 4
        self.check_health_every = 5  # Check every 5 steps
        
        # Restart button positions
        self.abandon_button = (630, 1099)
        self.restart_button = (589, 1074)
        self.new_seed_button = (1068, 828)
        
        # Input controllers
        self.keyboard = Controller()
        self.mouse_sensitivity = mouse_sensitivity
        
        # Track control states
        self.keys_pressed = set()
        
        # Reward tracking
        self.previous_rank = 'E'
        self.rank_values = {'E': 0, 'D': 1, 'C': 2, 'B': 4, 'A': 5, 'S': 6}
        self.episode_reward = 0
        
        # Load rank templates
        self.rank_templates = {}
        if os.path.exists('templates'):
            for rank in ['E', 'D', 'C', 'B', 'A', 'S']:
                template_path = f"templates/rank_{rank}.png"
                if os.path.exists(template_path):
                    self.rank_templates[rank] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        # Load lives templates
        self.lives_templates = {}
        for lives in [1, 2, 3, 4]:
            template_path = f"templates/lives_{lives}.png"
            if os.path.exists(template_path):
                self.lives_templates[lives] = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        
        # Death detection
        self.steps_since_rank_visible = 0
        self.max_steps_without_rank = 150
        self.current_lives = 4
        self.check_lives_every = 5
        
        # State
        self.current_step = 0
        self.max_steps = 2000
        self.total_episodes = 0
        
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_rank = 'E'
        self.steps_since_rank_visible = 0
        self.total_episodes += 1
        self.episode_reward = 0
        self.current_lives = 4
        self.previous_health_quarters = 4
        
        self._release_all_keys()
        
        # Auto-restart if not first episode
        if self.total_episodes > 1:
            self._auto_restart()
        

        
        obs = self._get_observation()
        return obs, {}
    
    def step(self, action):
        """Execute one step."""
        # Parse flattened action
        movement = int(action[0])
        mouse_x = float(action[1])
        mouse_y = float(action[2])
        fast_fall = int(action[3])
        speed_boost = int(action[4])
        
        self._take_action(movement, mouse_x, mouse_y, fast_fall, speed_boost)
        time.sleep(0.05)
        
        obs = self._get_observation()
        reward = self._calculate_reward()
        
        self.current_step += 1
        self.episode_reward += reward
        
        # Check health periodically
        if self.current_step % self.check_health_every == 0:
            current_health = self._read_health_quarters()
            
            # Penalize for losing health quarters
            if current_health < self.previous_health_quarters:
                quarters_lost = self.previous_health_quarters - current_health
                damage_penalty = -5.0 * quarters_lost
                reward += damage_penalty
                print(f"took damage: lost {quarters_lost} health quarter(s) - penalty: {damage_penalty:.2f}")
            
            self.previous_health_quarters = current_health
        
        # Check lives
        if self.current_step % self.check_lives_every == 0:
            self.current_lives = self._read_lives()
            if self.current_lives == 1:
                print("down to 1 life - restarting")
                terminated = True
                truncated = False
                
                death_penalty = -30.0 - (self.episode_reward * 0.5)
                reward += death_penalty
                
                return obs, reward, terminated, truncated, {}
        
        # Check for game over (death screen or rank missing)
        terminated = self._is_game_over()
        truncated = self.current_step >= self.max_steps
        
        # Apply death penalty if terminated
        if terminated:
            print("bleh")
            death_penalty = -30.0 - (self.episode_reward * 0.5)
            reward += death_penalty
        
        return obs, reward, terminated, truncated, {}
    
    def _get_observation(self):
        """Capture and process screen."""
        screenshot = self.sct.grab(self.monitor)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        img = cv2.resize(img, (128, 128))
        return img.reshape(128, 128, 1)
    
    def _extract_red(self, img):
        """Extract red pixels from BGR image."""
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Red color range
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        result = np.zeros(img.shape[:2], dtype=np.uint8)
        result[red_mask > 0] = 255
        
        return result
    
    def _read_lives(self):
        """Read current lives using template matching."""
        if not self.lives_templates:
            return 4
        
        try:
            screenshot = self.sct.grab(self.lives_region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Extract red hearts
            red_only = self._extract_red(img)
            
            # Match against templates
            best_match_score = -1
            best_lives = 4
            
            for lives, template in self.lives_templates.items():
                result = cv2.matchTemplate(red_only, template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                
                if max_val > best_match_score:
                    best_match_score = max_val
                    best_lives = lives
            
            if best_match_score > 0.6:
                return best_lives
                
        except Exception as e:
            pass
        
        return self.current_lives
    

    def _read_health_quarters(self):
        """Read health by counting red quarters in health bar."""
        try:
            screenshot = self.sct.grab(self.health_region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            
            # Extract red pixels (same as hearts)
            red_only = self._extract_red(img)
            
            # Divide into 4 quarters and count which have red
            width = red_only.shape[1]
            quarter_width = width // 4
            
            quarters_filled = 0
            for i in range(4):
                start_x = i * quarter_width
                end_x = start_x + quarter_width
                quarter = red_only[:, start_x:end_x]
                
                # If quarter has significant red pixels, count it as filled
                red_pixel_count = np.sum(quarter > 0)
                total_pixels = quarter.shape[0] * quarter.shape[1]
                
                if red_pixel_count > (total_pixels * 0.3):  # 30% threshold
                    quarters_filled += 1
            
            return quarters_filled
            
        except Exception as e:
            return self.previous_health_quarters
    
    def _is_death_screen(self):
        """Detect if on death/game over screen by checking whole screen darkness."""
        try:
            screenshot = self.sct.grab(self.monitor)
            img = np.array(screenshot)
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            
            # Death screen is mostly dark with diagonal lines
            avg_brightness = np.mean(img_gray)
            std_dev = np.std(img_gray)
            
            # Death screen characteristics
            if avg_brightness < 30 and std_dev < 20:
                return True
                
            return False
            
        except Exception as e:
            return False
    
    def _auto_restart(self):
        """Automatically restart level with new seed."""
        print("restarting level with new seed")
        
        # Press ESC
        self.keyboard.press(Key.esc)
        time.sleep(0.1)
        self.keyboard.release(Key.esc)
        time.sleep(0.8)
        
        # Click abandon shard
        pyautogui.click(self.abandon_button[0], self.abandon_button[1])
        time.sleep(0.8)
        
        # Click restart
        pyautogui.click(self.restart_button[0], self.restart_button[1])
        time.sleep(0.8)
        
        # Click new seed
        pyautogui.click(self.new_seed_button[0], self.new_seed_button[1])
        time.sleep(1.333)
        
        print("level restarted")
    
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
            return 0.0
        
        reward = 0.01
        
        rank_value = self.rank_values.get(current_rank, 0)
        reward = rank_value * 1.0
        
        prev_rank_value = self.rank_values.get(self.previous_rank, 0)
        if rank_value > prev_rank_value:
            reward += 10.0
        elif rank_value < prev_rank_value:
            reward -= 2.0
        
        self.previous_rank = current_rank
        return reward
    
    def _is_game_over(self):
        """Detect if dead or level complete."""
        # Check for death screen first
        if self._is_death_screen():
            print("death screen detected")
            return True
        
        # Fallback: check if rank missing for too long
        if self.steps_since_rank_visible >= self.max_steps_without_rank:
            print("rank missing - assuming death")
            return True
        
        return False
    
    def _take_action(self, movement, mouse_x, mouse_y, fast_fall, speed_boost):
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
        
        # Handle fast fall (Space bar)
        if fast_fall == 1:
            if Key.space not in self.keys_pressed:
                self.keyboard.press(Key.space)
                self.keys_pressed.add(Key.space)
        else:
            if Key.space in self.keys_pressed:
                self.keyboard.release(Key.space)
                self.keys_pressed.discard(Key.space)
        
        # Handle speed boost (Left click)
        if speed_boost == 1:
            if 'left_click' not in self.keys_pressed:
                pydirectinput.mouseDown()
                self.keys_pressed.add('left_click')
        else:
            if 'left_click' in self.keys_pressed:
                pydirectinput.mouseUp()
                self.keys_pressed.discard('left_click')
        
        # Handle mouse movement
        mouse_dx = int(mouse_x * self.mouse_sensitivity)
        mouse_dy = int(mouse_y * self.mouse_sensitivity)
        
        if mouse_dx != 0 or mouse_dy != 0:
            pydirectinput.moveRel(mouse_dx, mouse_dy, relative=True)
    
    def _release_all_keys(self):
        """Release all pressed keys."""
        # Release WASD keys
        for key in list(self.keys_pressed):
            if key in ['w', 'a', 's', 'd']:
                self.keyboard.release(key)
        
        # Release space bar
        if Key.space in self.keys_pressed:
            self.keyboard.release(Key.space)
        
        # Release left click
        if 'left_click' in self.keys_pressed:
            pydirectinput.mouseUp()
        
        self.keys_pressed.clear()
    
    def close(self):
        """Cleanup."""
        self._release_all_keys()