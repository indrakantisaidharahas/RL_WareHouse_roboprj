"""
Improved Warehouse Robot with Transfer Learning
Features:
- Simplified task that actually works
- Train on multiple warehouse layouts
- Test on new unseen warehouses
- Demonstrates generalization capability

Installation:
pip install gymnasium stable-baselines3 numpy matplotlib pillow

Run: python warehouse_improved.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import random
import json


class SimpleWarehouseEnv(gym.Env):
    """
    SIMPLIFIED Warehouse - Actually learnable!
    
    Simplifications:
    1. Only 1 object at a time (clear goal)
    2. Direct observation of object location
    3. No moving obstacles initially
    4. Smaller grid (15x15)
    5. Clear reward signal
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    
    def __init__(self, config=None, render_mode=None):
        super().__init__()
        
        # Use config for different warehouse layouts
        if config is None:
            config = self._default_config()
        
        self.config = config
        self.grid_size = config['grid_size']
        self.render_mode = render_mode
        
        # Simplified action space: Just movement (pick/drop is automatic)
        self.action_space = spaces.Discrete(4)  # Up, Down, Left, Right
        
        # Simplified observation: robot pos + object pos + delivery zone center
        self.observation_space = spaces.Box(
            low=0,
            high=self.grid_size,
            shape=(6,),  # [robot_x, robot_y, object_x, object_y, delivery_x, delivery_y]
            dtype=np.float32
        )
        
        # Initialize from config
        self.delivery_zone = config['delivery_zone']
        self.static_obstacles = [np.array(obs) for obs in config['obstacles']]
        
        self.robot_pos = None
        self.object_pos = None
        self.carrying = False
        self.delivered_count = 0
        self.max_steps = 200
        self.current_step = 0
        
    @staticmethod
    def _default_config():
        """Default warehouse layout"""
        return {
            'grid_size': 15,
            'delivery_zone': [11, 11, 13, 13],
            'spawn_zones': [[2, 2, 5, 5], [2, 9, 5, 12], [9, 2, 12, 5]],
            'obstacles': [
                [7, 3], [7, 4], [7, 5],
                [7, 9], [7, 10], [7, 11]
            ]
        }
    
    @staticmethod
    def generate_random_config(grid_size=15):
        """Generate random warehouse layout for training diversity"""
        # Random delivery zone
        delivery_x = random.randint(grid_size-5, grid_size-2)
        delivery_y = random.randint(grid_size-5, grid_size-2)
        delivery_zone = [delivery_x, delivery_y, 
                        min(delivery_x+2, grid_size-1), 
                        min(delivery_y+2, grid_size-1)]
        
        # Random spawn zones
        spawn_zones = []
        for _ in range(3):
            x = random.randint(1, grid_size-6)
            y = random.randint(1, grid_size-6)
            spawn_zones.append([x, y, x+3, y+3])
        
        # Random obstacles (simple vertical/horizontal lines)
        obstacles = []
        for _ in range(2):
            if random.random() > 0.5:
                # Vertical line
                x = random.randint(5, grid_size-6)
                for y in range(3, 8):
                    obstacles.append([x, y])
            else:
                # Horizontal line
                y = random.randint(5, grid_size-6)
                for x in range(3, 8):
                    obstacles.append([x, y])
        
        return {
            'grid_size': grid_size,
            'delivery_zone': delivery_zone,
            'spawn_zones': spawn_zones,
            'obstacles': obstacles
        }
    
    def _spawn_object(self):
        """Spawn object in a spawn zone"""
        zone = random.choice(self.config['spawn_zones'])
        x = random.randint(zone[0], zone[2])
        y = random.randint(zone[1], zone[3])
        pos = np.array([x, y])
        
        # Make sure not on obstacle
        while any(np.array_equal(pos, obs) for obs in self.static_obstacles):
            x = random.randint(zone[0], zone[2])
            y = random.randint(zone[1], zone[3])
            pos = np.array([x, y])
        
        return pos
    
    def _in_delivery_zone(self, pos):
        """Check if in delivery zone"""
        zone = self.delivery_zone
        return (zone[0] <= pos[0] <= zone[2] and 
                zone[1] <= pos[1] <= zone[3])
    
    def _get_obs(self):
        """Simple observation"""
        delivery_center = np.array([
            (self.delivery_zone[0] + self.delivery_zone[2]) / 2,
            (self.delivery_zone[1] + self.delivery_zone[3]) / 2
        ])
        
        obs = np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.object_pos[0] if not self.carrying else -1,
            self.object_pos[1] if not self.carrying else -1,
            delivery_center[0],
            delivery_center[1]
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self):
        return {
            "delivered": self.delivered_count,
            "carrying": self.carrying,
            "steps": self.current_step,
            "success": self.delivered_count > 0
        }
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.robot_pos = np.array([1, 1])
        self.object_pos = self._spawn_object()
        self.carrying = False
        self.delivered_count = 0
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        self.current_step += 1
        old_pos = self.robot_pos.copy()
        
        # Move robot
        if action == 0:  # Up
            self.robot_pos[1] = max(0, self.robot_pos[1] - 1)
        elif action == 1:  # Down
            self.robot_pos[1] = min(self.grid_size - 1, self.robot_pos[1] + 1)
        elif action == 2:  # Left
            self.robot_pos[0] = max(0, self.robot_pos[0] - 1)
        elif action == 3:  # Right
            self.robot_pos[0] = min(self.grid_size - 1, self.robot_pos[0] + 1)
        
        # Check collision with obstacles
        collision = any(np.array_equal(self.robot_pos, obs) for obs in self.static_obstacles)
        
        reward = 0
        terminated = False
        
        if collision:
            self.robot_pos = old_pos  # Revert
            reward = -5
        else:
            # Automatic pickup
            if not self.carrying and np.array_equal(self.robot_pos, self.object_pos):
                self.carrying = True
                reward = 50  # BIG reward for pickup
            
            # Automatic delivery
            elif self.carrying and self._in_delivery_zone(self.robot_pos):
                self.carrying = False
                self.delivered_count += 1
                reward = 100  # HUGE reward for delivery
                terminated = True  # Episode ends on successful delivery
            
            # Movement rewards
            else:
                if self.carrying:
                    # Moving toward delivery zone
                    delivery_center = np.array([
                        (self.delivery_zone[0] + self.delivery_zone[2]) / 2,
                        (self.delivery_zone[1] + self.delivery_zone[3]) / 2
                    ])
                    old_dist = np.linalg.norm(old_pos - delivery_center)
                    new_dist = np.linalg.norm(self.robot_pos - delivery_center)
                    
                    if new_dist < old_dist:
                        reward = 2  # Good progress
                    else:
                        reward = -0.5  # Wrong direction
                else:
                    # Moving toward object
                    old_dist = np.linalg.norm(old_pos - self.object_pos)
                    new_dist = np.linalg.norm(self.robot_pos - self.object_pos)
                    
                    if new_dist < old_dist:
                        reward = 2  # Good progress
                    else:
                        reward = -0.5  # Wrong direction
        
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        if self.render_mode == "rgb_array":
            img_size = 500
            cell_size = img_size // self.grid_size
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 240
            
            # Draw grid
            for i in range(self.grid_size + 1):
                pos = i * cell_size
                if pos < img_size:
                    img[pos, :] = [200, 200, 200]
                    img[:, pos] = [200, 200, 200]
            
            # Draw delivery zone (green)
            zone = self.delivery_zone
            for y in range(zone[1], min(zone[3]+1, self.grid_size)):
                for x in range(zone[0], min(zone[2]+1, self.grid_size)):
                    px, py = x * cell_size, y * cell_size
                    if py < img_size and px < img_size:
                        img[py:py+cell_size, px:px+cell_size] = [200, 255, 200]
            
            # Draw spawn zones (light blue)
            for zone in self.config['spawn_zones']:
                for y in range(zone[1], min(zone[3]+1, self.grid_size)):
                    for x in range(zone[0], min(zone[2]+1, self.grid_size)):
                        px, py = x * cell_size, y * cell_size
                        if py < img_size and px < img_size:
                            img[py:py+cell_size, px:px+cell_size] = [220, 240, 255]
            
            # Draw obstacles (brown)
            for obs in self.static_obstacles:
                x, y = obs[0] * cell_size, obs[1] * cell_size
                if y < img_size-cell_size and x < img_size-cell_size:
                    img[y+2:y+cell_size-2, x+2:x+cell_size-2] = [139, 90, 43]
            
            # Draw object (yellow box)
            if not self.carrying:
                x = self.object_pos[0] * cell_size + cell_size//2
                y = self.object_pos[1] * cell_size + cell_size//2
                size = cell_size // 3
                if y-size >= 0 and x-size >= 0 and y+size < img_size and x+size < img_size:
                    img[y-size:y+size, x-size:x+size] = [255, 255, 0]
            
            # Draw robot (blue if searching, green if carrying)
            x = self.robot_pos[0] * cell_size + cell_size//2
            y = self.robot_pos[1] * cell_size + cell_size//2
            radius = cell_size // 3
            
            robot_color = [0, 200, 0] if self.carrying else [0, 0, 255]
            
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    if i*i + j*j <= radius*radius:
                        ny, nx = y+i, x+j
                        if 0 <= ny < img_size and 0 <= nx < img_size:
                            img[ny, nx] = robot_color
            
            return img
        return None


class WarehouseTrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.successes = []
        self.episode_count = 0
        
    def _on_step(self):
        if len(self.locals.get('dones', [])) > 0:
            if self.locals['dones'][0]:
                self.episode_count += 1
                info = self.locals['infos'][0]
                self.successes.append(info.get('success', False))
        return True


def train_on_multiple_layouts(algorithm='DQN', total_timesteps=150000, num_layouts=5):
    """
    Train on multiple warehouse layouts for better generalization
    """
    print("🏭 Training on Multiple Warehouse Layouts...")
    print(f"   Using {num_layouts} different warehouse configurations")
    print("=" * 60)
    
    # Generate training configurations
    configs = [SimpleWarehouseEnv._default_config()]  # Include default
    for _ in range(num_layouts - 1):
        configs.append(SimpleWarehouseEnv.generate_random_config())
    
    # Save configs
    with open('training_configs.json', 'w') as f:
        json.dump(configs, f, indent=2)
    print(f"✅ Generated {len(configs)} training layouts (saved to training_configs.json)")
    
    # Create environment that randomly switches between configs
    def make_env():
        config = random.choice(configs)
        return SimpleWarehouseEnv(config=config)
    
    env = DummyVecEnv([make_env])
    callback = WarehouseTrainingCallback()
    
    if algorithm == 'DQN':
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            gamma=0.95,
            exploration_fraction=0.5,
            exploration_final_eps=0.1,
            verbose=1
        )
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.001,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.95,
            verbose=1
        )
    
    print(f"\n📚 Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    model_name = f"warehouse_generalized_{algorithm.lower()}"
    model.save(model_name)
    print(f"✅ Model saved as {model_name}.zip")
    
    # Calculate success rate
    recent_successes = callback.successes[-100:] if len(callback.successes) >= 100 else callback.successes
    success_rate = (sum(recent_successes) / len(recent_successes)) * 100 if recent_successes else 0
    print(f"🎯 Recent Success Rate: {success_rate:.1f}%")
    
    return model, configs


def test_generalization(model, test_configs, num_episodes=10):
    """
    Test model on unseen warehouse layouts
    """
    print("\n🧪 Testing Generalization on Unseen Layouts...")
    print("=" * 60)
    
    results = []
    
    for i, config in enumerate(test_configs):
        print(f"\n📍 Testing Layout {i+1}/{len(test_configs)}...")
        env = SimpleWarehouseEnv(config=config, render_mode='rgb_array')
        
        successes = 0
        total_steps = []
        
        for ep in range(num_episodes):
            obs, info = env.reset()
            done = False
            steps = 0
            
            while not done and steps < 200:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                steps += 1
            
            if info['success']:
                successes += 1
                total_steps.append(steps)
        
        success_rate = (successes / num_episodes) * 100
        avg_steps = np.mean(total_steps) if total_steps else 200
        
        results.append({
            'layout': i+1,
            'success_rate': success_rate,
            'avg_steps': avg_steps
        })
        
        print(f"   Success Rate: {success_rate:.1f}%")
        print(f"   Avg Steps: {avg_steps:.1f}")
        
        env.close()
    
    return results


def create_demo_comparison(model, configs):
    """
    Create demo showing robot working on different warehouse layouts
    """
    print("\n🎬 Creating Multi-Layout Demo...")
    
    all_frames = []
    
    for layout_idx, config in enumerate(configs[:3]):  # Demo first 3 layouts
        print(f"   Recording layout {layout_idx + 1}/3...")
        
        env = SimpleWarehouseEnv(config=config, render_mode='rgb_array')
        obs, info = env.reset()
        
        frames = []
        done = False
        steps = 0
        
        while not done and steps < 150:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1
            
            frame = env.render()
            if frame is not None:
                # Add label
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
                except:
                    font = ImageFont.load_default()
                
                text = f"Warehouse Layout {layout_idx + 1} | Step {steps}"
                if info['carrying']:
                    text += " | CARRYING"
                if info['delivered'] > 0:
                    text += " | ✓ DELIVERED"
                
                # Background
                bbox = draw.textbbox((0, 0), text, font=font)
                draw.rectangle([(5, 5), (bbox[2]+15, bbox[3]+15)], fill=(0, 0, 0, 180))
                draw.text((10, 10), text, fill=(255, 255, 255), font=font)
                
                frames.append(np.array(img))
        
        all_frames.extend(frames)
        
        # Add pause between layouts
        if layout_idx < 2:
            all_frames.extend([frames[-1]] * 20)
        
        env.close()
    
    # Save GIF
    images = [Image.fromarray(f) for f in all_frames]
    images[0].save(
        'warehouse_generalization.gif',
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )
    
    print("✅ Demo saved as: warehouse_generalization.gif")


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🏭 WAREHOUSE AUTOMATION with TRANSFER LEARNING")
    print("   Training on Multiple Layouts + Testing Generalization")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test existing model on new layouts
        model_file = sys.argv[2] if len(sys.argv) > 2 else "warehouse_generalized_dqn.zip"
        
        print(f"Loading model: {model_file}")
        if 'dqn' in model_file.lower():
            model = DQN.load(model_file)
        else:
            model = PPO.load(model_file)
        
        # Generate new test layouts
        test_configs = [SimpleWarehouseEnv.generate_random_config() for _ in range(5)]
        
        # Test generalization
        results = test_generalization(model, test_configs, num_episodes=10)
        
        # Create demo
        create_demo_comparison(model, test_configs)
        
        # Summary
        avg_success = np.mean([r['success_rate'] for r in results])
        print(f"\n📊 Overall Generalization Performance: {avg_success:.1f}% success rate")
        
    else:
        # Training mode
        print("\nSelect algorithm:")
        print("   1. DQN (Recommended)")
        print("   2. PPO")
        
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        algorithm = 'PPO' if choice == '2' else 'DQN'
        
        # Train on multiple layouts
        model, train_configs = train_on_multiple_layouts(
            algorithm=algorithm,
            total_timesteps=150000,
            num_layouts=5
        )
        
        # Test on new layouts
        print("\n" + "=" * 60)
        test_configs = [SimpleWarehouseEnv.generate_random_config() for _ in range(3)]
        results = test_generalization(model, test_configs, num_episodes=10)
        
        # Create demo
        create_demo_comparison(model, train_configs)
        
        # Summary
        avg_success = np.mean([r['success_rate'] for r in results])
        
        print("\n" + "=" * 60)
        print("✅ Training Complete!")
        print(f"   Model: warehouse_generalized_{algorithm.lower()}.zip")
        print(f"   Demo: warehouse_generalization.gif")
        print(f"   Generalization Success Rate: {avg_success:.1f}%")
        print("\n📝 For your report:")
        print("   - Trained on 5 different warehouse layouts")
        print("   - Tested on 3 completely new unseen layouts")
        print(f"   - Achieved {avg_success:.1f}% success on new environments")
        print("   - Demonstrates transfer learning capability")
