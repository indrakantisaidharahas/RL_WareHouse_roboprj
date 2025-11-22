import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import random
import os
from collections import deque


class FinalWarehouseEnv(gym.Env):
    """FINAL FIXED Warehouse Environment"""
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    
    def __init__(self, config=None, render_mode='rgb_array'):
        super().__init__()
        
        if config is None:
            config = self._create_layout()
        
        self.config = config
        self.grid_size = config['grid_size']
        self.render_mode = render_mode
        
        self.action_space = spaces.Discrete(4)
        
        # Enhanced observation with obstacle detection
        self.observation_space = spaces.Box(
            low=-1, high=self.grid_size, shape=(13,), dtype=np.float32
        )
        
        self.delivery_zone = config['delivery_zone']
        self.static_obstacles = [np.array(obs, dtype=float) for obs in config['obstacles']]
        self.pickup_radius = 1.5
        
        # State
        self.robot_pos = None
        self.object_pos = None
        self.carrying = False
        self.delivered_count = 0
        self.max_steps = 300
        self.current_step = 0
        self.collision_count = 0
        self.path_history = []
        
        # Anti-oscillation
        self.position_memory = deque(maxlen=20)
        self.action_memory = deque(maxlen=10)
        
        # Progress tracking
        self.best_dist_to_object = float('inf')
        self.best_dist_to_delivery = float('inf')
        self.steps_without_progress = 0
        
    @staticmethod
    def _create_layout(grid_size=20):
        """Organized warehouse with clear corridors"""
        delivery_zone = [grid_size-4, grid_size-4, grid_size-1, grid_size-1]
        
        spawn_zones = [
            [1, 1, 4, 4],
            [1, grid_size-5, 4, grid_size-2],
            [grid_size-5, 1, grid_size-2, 4]
        ]
        
        # Vertical shelves with horizontal corridors
        obstacles = []
        shelf_columns = [5, 9, 13]
        for col in shelf_columns:
            for row in range(3, grid_size - 4):
                if row % 4 != 0:  # Leave gaps
                    obstacles.append([col, row])
        
        # Horizontal shelf
        for col in range(6, grid_size - 5):
            if col not in shelf_columns:
                obstacles.append([col, grid_size // 2])
        
        return {
            'grid_size': grid_size,
            'delivery_zone': delivery_zone,
            'spawn_zones': spawn_zones,
            'obstacles': obstacles
        }
    
    def _spawn_object(self):
        """Spawn object in spawn zone"""
        zone = random.choice(self.config['spawn_zones'])
        for _ in range(50):
            x = random.randint(zone[0], zone[2])
            y = random.randint(zone[1], zone[3])
            pos = np.array([x, y], dtype=float)
            if not any(np.array_equal(pos, obs) for obs in self.static_obstacles):
                return pos
        return np.array([zone[0], zone[1]], dtype=float)
    
    def _in_delivery_zone(self, pos):
        zone = self.delivery_zone
        return (zone[0] <= pos[0] <= zone[2] and zone[1] <= pos[1] <= zone[3])
    
    def _get_delivery_center(self):
        return np.array([
            (self.delivery_zone[0] + self.delivery_zone[2]) / 2,
            (self.delivery_zone[1] + self.delivery_zone[3]) / 2
        ])
    
    def _check_obstacle_at(self, pos):
        """Check if position has obstacle"""
        for obs in self.static_obstacles:
            if np.allclose(pos, obs, atol=0.1):
                return True
        return False
    
    def _get_obstacle_info(self):
        """Get obstacle presence in 4 directions - FIXED"""
        obstacles = {}
        
        # Up
        test_pos = self.robot_pos.copy()
        test_pos[1] -= 1
        obstacles['up'] = 1 if (test_pos[1] < 0 or self._check_obstacle_at(test_pos)) else 0
        
        # Down
        test_pos = self.robot_pos.copy()
        test_pos[1] += 1
        obstacles['down'] = 1 if (test_pos[1] >= self.grid_size or self._check_obstacle_at(test_pos)) else 0
        
        # Left
        test_pos = self.robot_pos.copy()
        test_pos[0] -= 1
        obstacles['left'] = 1 if (test_pos[0] < 0 or self._check_obstacle_at(test_pos)) else 0
        
        # Right
        test_pos = self.robot_pos.copy()
        test_pos[0] += 1
        obstacles['right'] = 1 if (test_pos[0] >= self.grid_size or self._check_obstacle_at(test_pos)) else 0
        
        return obstacles
    
    def _detect_stuck(self):
        """Detect oscillation - FIXED for numpy arrays"""
        if len(self.position_memory) < 15:
            return False
        
        recent = list(self.position_memory)[-15:]
        # Convert to tuples for set comparison
        unique = len(set(tuple(p) for p in recent))
        return unique <= 4
    
    def _get_obs(self):
        """Enhanced observation"""
        delivery_center = self._get_delivery_center()
        
        if not self.carrying:
            dist_to_object = np.linalg.norm(self.robot_pos - self.object_pos)
            dist_to_delivery = 0
        else:
            dist_to_object = 0
            dist_to_delivery = np.linalg.norm(self.robot_pos - delivery_center)
        
        obstacles = self._get_obstacle_info()
        
        return np.array([
            self.robot_pos[0] / self.grid_size,
            self.robot_pos[1] / self.grid_size,
            self.object_pos[0] / self.grid_size if not self.carrying else -1,
            self.object_pos[1] / self.grid_size if not self.carrying else -1,
            delivery_center[0] / self.grid_size,
            delivery_center[1] / self.grid_size,
            dist_to_object / self.grid_size,
            dist_to_delivery / self.grid_size,
            1.0 if self.carrying else 0.0,
            obstacles['up'],
            obstacles['down'],
            obstacles['left'],
            obstacles['right']
        ], dtype=np.float32)
    
    def _get_info(self):
        return {
            "delivered": self.delivered_count,
            "carrying": self.carrying,
            "steps": self.current_step,
            "success": self.delivered_count > 0,
            "collision": self.collision_count,
            "path_length": len(self.path_history)
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.robot_pos = np.array([2, 2], dtype=float)
        self.object_pos = self._spawn_object()
        self.carrying = False
        self.delivered_count = 0
        self.current_step = 0
        self.collision_count = 0
        
        self.path_history = [self.robot_pos.copy()]
        self.position_memory.clear()
        self.action_memory.clear()
        
        self.best_dist_to_object = np.linalg.norm(self.robot_pos - self.object_pos)
        self.best_dist_to_delivery = float('inf')
        self.steps_without_progress = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.current_step += 1
        old_pos = self.robot_pos.copy()
        
        # Track position and action - FIXED numpy array issue
        self.position_memory.append(self.robot_pos.copy())
        action_int = int(action) if hasattr(action, '__iter__') else action
        self.action_memory.append(action_int)
        
        # Move robot
        if action_int == 0:    # Up
            self.robot_pos[1] = max(0, self.robot_pos[1] - 1)
        elif action_int == 1:  # Down
            self.robot_pos[1] = min(self.grid_size - 1, self.robot_pos[1] + 1)
        elif action_int == 2:  # Left
            self.robot_pos[0] = max(0, self.robot_pos[0] - 1)
        elif action_int == 3:  # Right
            self.robot_pos[0] = min(self.grid_size - 1, self.robot_pos[0] + 1)
        
        self.path_history.append(self.robot_pos.copy())
        
        # Check collision
        collision = self._check_obstacle_at(self.robot_pos)
        
        reward = 0
        terminated = False
        
        # STRONG collision penalty
        if collision:
            self.robot_pos = old_pos
            reward = -100  # VERY STRONG
            self.collision_count += 1
            self.steps_without_progress += 1
        
        elif np.array_equal(self.robot_pos, old_pos):
            reward = -10  # Boundary hit
            self.steps_without_progress += 1
        
        else:
            # Valid movement - PROPER reward shaping
            
            if not self.carrying:
                dist = np.linalg.norm(self.robot_pos - self.object_pos)
                
                # Pickup
                if dist <= self.pickup_radius:
                    self.carrying = True
                    reward = 200
                    self.steps_without_progress = 0
                    print(f"   ✓ PICKUP at step {self.current_step}")
                
                else:
                    old_dist = np.linalg.norm(old_pos - self.object_pos)
                    
                    if dist < self.best_dist_to_object:
                        self.best_dist_to_object = dist
                        reward = 10
                        self.steps_without_progress = 0
                    elif dist < old_dist:
                        reward = 3
                        self.steps_without_progress = 0
                    else:
                        reward = -3
                        self.steps_without_progress += 1
            
            else:
                # Carrying
                if self._in_delivery_zone(self.robot_pos):
                    self.carrying = False
                    self.delivered_count += 1
                    time_bonus = max(0, 150 - self.current_step)
                    reward = 500 + time_bonus
                    terminated = True
                    print(f"   ★ DELIVERED at step {self.current_step}!")
                
                else:
                    delivery_center = self._get_delivery_center()
                    dist = np.linalg.norm(self.robot_pos - delivery_center)
                    old_dist = np.linalg.norm(old_pos - delivery_center)
                    
                    if dist < self.best_dist_to_delivery:
                        self.best_dist_to_delivery = dist
                        reward = 10
                        self.steps_without_progress = 0
                    elif dist < old_dist:
                        reward = 3
                        self.steps_without_progress = 0
                    else:
                        reward = -3
                        self.steps_without_progress += 1
            
            # Anti-oscillation
            if self._detect_stuck():
                reward -= 50
        
        # Time penalty
        reward -= 0.1
        
        # Stuck termination
        if self.steps_without_progress > 80:
            reward -= 100
            terminated = True
        
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render with PIL - NO cv2 needed"""
        if self.render_mode is None:
            return None
        
        img_size = 800
        cell_size = img_size // self.grid_size
        
        img = Image.new('RGB', (img_size, img_size), color=(245, 245, 245))
        draw = ImageDraw.Draw(img)
        
        # Grid
        for i in range(self.grid_size + 1):
            pos = i * cell_size
            draw.line([(pos, 0), (pos, img_size)], fill=(220, 220, 220))
            draw.line([(0, pos), (img_size, pos)], fill=(220, 220, 220))
        
        # Delivery zone (green)
        zone = self.delivery_zone
        for y in range(int(zone[1]), int(zone[3])+1):
            for x in range(int(zone[0]), int(zone[2])+1):
                px, py = x * cell_size, y * cell_size
                draw.rectangle([px+1, py+1, px+cell_size-1, py+cell_size-1], 
                              fill=(180, 255, 180))
        
        # Spawn zones (blue)
        for zone in self.config['spawn_zones']:
            for y in range(zone[1], zone[3]+1):
                for x in range(zone[0], zone[2]+1):
                    px, py = x * cell_size, y * cell_size
                    draw.rectangle([px+1, py+1, px+cell_size-1, py+cell_size-1],
                                  fill=(255, 240, 200))
        
        # Obstacles (brown)
        for obs in self.static_obstacles:
            x, y = int(obs[0]) * cell_size, int(obs[1]) * cell_size
            draw.rectangle([x+3, y+3, x+cell_size-3, y+cell_size-3], fill=(101, 67, 33))
        
        # Path (colored trail)
        if len(self.path_history) > 1:
            for i in range(1, min(len(self.path_history), 100)):  # Last 100 steps
                p1 = self.path_history[i-1]
                p2 = self.path_history[i]
                
                x1 = int(p1[0] * cell_size + cell_size // 2)
                y1 = int(p1[1] * cell_size + cell_size // 2)
                x2 = int(p2[0] * cell_size + cell_size // 2)
                y2 = int(p2[1] * cell_size + cell_size // 2)
                
                color = (0, 150, 255) if self.carrying else (255, 140, 0)
                draw.line([(x1, y1), (x2, y2)], fill=color, width=3)
        
        # Object (yellow)
        if not self.carrying:
            x = int(self.object_pos[0] * cell_size + cell_size // 2)
            y = int(self.object_pos[1] * cell_size + cell_size // 2)
            size = cell_size // 3
            draw.rectangle([x-size, y-size, x+size, y+size], fill=(0, 215, 255))
        
        # Robot (green if carrying, orange if not)
        x = int(self.robot_pos[0] * cell_size + cell_size // 2)
        y = int(self.robot_pos[1] * cell_size + cell_size // 2)
        radius = cell_size // 3
        
        color = (0, 200, 0) if self.carrying else (255, 100, 0)
        draw.ellipse([x-radius, y-radius, x+radius, y+radius], fill=color, outline=(0, 0, 0), width=2)
        
        # Info text
        try:
            font = ImageFont.truetype("arial.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        status = "CARRYING" if self.carrying else "SEARCHING"
        draw.text((10, 10), f"Step: {self.current_step} | {status} | Collisions: {self.collision_count}", 
                 fill=(0, 0, 0), font=font)
        
        return np.array(img)


class SimpleCallback(BaseCallback):
    """Simple callback without crashes"""
    
    def __init__(self):
        super().__init__()
        self.successes = []
        self.collisions = []
        self.episodes = 0
        
    def _on_step(self):
        if len(self.locals.get('dones', [])) > 0 and self.locals['dones'][0]:
            info = self.locals['infos'][0]
            self.episodes += 1
            
            self.successes.append(1 if info.get('success', False) else 0)
            self.collisions.append(info.get('collision', 0))
            
            if len(self.successes) >= 100:
                success_rate = np.mean(self.successes[-100:])
                avg_collision = np.mean(self.collisions[-100:])
                
                self.logger.record("metrics/success_rate", success_rate * 100)
                self.logger.record("metrics/avg_collisions", avg_collision)
                
                if self.episodes % 100 == 0:
                    print(f"\n   📊 Ep {self.episodes}: Success={success_rate*100:.1f}%, Collisions={avg_collision:.1f}")
        
        return True


def train_final():
    """Train with optimized PPO"""
    print("=" * 80)
    print("🏭 FINAL WAREHOUSE ROBOT TRAINING")
    print("=" * 80)
    
    env = DummyVecEnv([lambda: FinalWarehouseEnv()])
    callback = SimpleCallback()
    
    log_dir = "./logs_final/"
    os.makedirs(log_dir, exist_ok=True)
    
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=log_dir,
        verbose=1
    )
    
    print("\n📚 Training for 500,000 steps...")
    print(f"   TensorBoard: tensorboard --logdir {log_dir}")
    print("=" * 80)
    
    model.learn(
        total_timesteps=500000,
        callback=callback,
        tb_log_name="PPO_final",
        progress_bar=True
    )
    
    model.save("warehouse_final")
    
    success_rate = np.mean(callback.successes[-100:]) if len(callback.successes) >= 100 else 0
    avg_collision = np.mean(callback.collisions[-100:]) if len(callback.collisions) >= 100 else 0
    
    print(f"\n✅ Training Complete!")
    print(f"   Model: warehouse_final.zip")
    print(f"   Success Rate: {success_rate*100:.1f}%")
    print(f"   Avg Collisions: {avg_collision:.1f}")
    
    return model


def test_final(model_path="warehouse_final.zip", episodes=10):
    """Test final model with automatic video generation"""
    print(f"\n🧪 Testing: {model_path}")
    print("=" * 80)
    
    if not os.path.exists(model_path):
        print(f"❌ Model not found!")
        return
    
    model = PPO.load(model_path)
    env = FinalWarehouseEnv()
    
    successes = 0
    total_steps = []
    total_collisions = []
    
    # ALWAYS save video frames for first 3 episodes
    video_frames = []
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        steps = 0
        
        # Record first 3 episodes
        record = (ep < 3)
        
        while not done and steps < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            # Capture frame
            if record:
                frame = env.render()
                if frame is not None:
                    video_frames.append(Image.fromarray(frame))
        
        if info['success']:
            successes += 1
            total_steps.append(steps)
        
        total_collisions.append(info['collision'])
        
        result = "✓ SUCCESS" if info['success'] else "✗ FAILED"
        print(f"   Ep {ep+1}: {result} | Steps: {steps} | Collisions: {info['collision']}")
    
# AUTOMATICALLY save GIF with unique numbering
    if len(video_frames) > 0:
    # Find next available test number
        test_num = 1
        while os.path.exists(f"warehouse_test_{test_num}.gif"):
            test_num += 1
    
        video_path = f"warehouse_test_{test_num}.gif"
        print(f"\n🎬 Saving video...")
        video_frames[0].save(
        video_path,
        save_all=True,
        append_images=video_frames[1:],
        duration=80,
        loop=0
    )
    print(f"✅ Video saved: {video_path}")




if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 80)
    print("🏭 FINAL WORKING WAREHOUSE ROBOT")
    print("=" * 80)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        test_final()
    else:
        train_final()
        print("\n💡 Now test with: python 1.py test")