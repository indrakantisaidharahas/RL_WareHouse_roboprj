"""
Robot Path Planning in DYNAMIC Environments
Features: Moving obstacles, appearing/disappearing obstacles (producer-consumer)

Installation:
pip install gymnasium stable-baselines3 numpy matplotlib pillow

Run: python dynamic_environment.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
import random


class DynamicRobotEnv(gym.Env):
    """
    Dynamic Environment with Moving Obstacles
    
    Features:
    - Moving obstacles (patrol patterns)
    - Appearing/disappearing obstacles (producer-consumer model)
    - Real-time adaptation required
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=15, num_static_obstacles=10, num_moving_obstacles=5, 
                 enable_producer_consumer=True, render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles
        self.enable_producer_consumer = enable_producer_consumer
        self.render_mode = render_mode
        
        # Action and observation space
        self.action_space = spaces.Discrete(4)
        
        # Observation: robot pos, goal pos, nearby obstacles
        obs_size = 4 + (9 * 2)  # robot(2) + goal(2) + 9 cells around robot (x,y each)
        self.observation_space = spaces.Box(
            low=-1, 
            high=grid_size, 
            shape=(obs_size,), 
            dtype=np.float32
        )
        
        # Initialize
        self.robot_pos = np.array([1, 1])
        self.goal_pos = np.array([grid_size-2, grid_size-2])
        self.static_obstacles = []
        self.moving_obstacles = []
        self.dynamic_obstacles = []  # Producer-consumer obstacles
        self.max_steps = 200
        self.current_step = 0
        self.producer_consumer_timer = 0
        
    def _generate_obstacles(self):
        """Generate static and moving obstacles"""
        # Static obstacles
        self.static_obstacles = []
        while len(self.static_obstacles) < self.num_static_obstacles:
            pos = np.random.randint(2, self.grid_size-2, size=2)
            if (not np.array_equal(pos, [1, 1]) and 
                not np.array_equal(pos, self.goal_pos)):
                if not any(np.array_equal(pos, obs) for obs in self.static_obstacles):
                    self.static_obstacles.append(pos)
        
        # Moving obstacles with velocity and direction
        self.moving_obstacles = []
        for _ in range(self.num_moving_obstacles):
            pos = np.random.randint(3, self.grid_size-3, size=2)
            direction = random.choice(['horizontal', 'vertical', 'diagonal'])
            velocity = random.choice([-1, 1])
            self.moving_obstacles.append({
                'pos': pos,
                'direction': direction,
                'velocity': velocity,
                'start_pos': pos.copy()
            })
        
        # Initialize dynamic obstacles
        self.dynamic_obstacles = []
    
    def _update_moving_obstacles(self):
        """Update positions of moving obstacles"""
        for obs in self.moving_obstacles:
            pos = obs['pos']
            direction = obs['direction']
            velocity = obs['velocity']
            
            # Move based on direction
            if direction == 'horizontal':
                pos[0] += velocity
                # Bounce at boundaries
                if pos[0] <= 1 or pos[0] >= self.grid_size - 2:
                    obs['velocity'] *= -1
                    pos[0] = np.clip(pos[0], 2, self.grid_size - 3)
            
            elif direction == 'vertical':
                pos[1] += velocity
                if pos[1] <= 1 or pos[1] >= self.grid_size - 2:
                    obs['velocity'] *= -1
                    pos[1] = np.clip(pos[1], 2, self.grid_size - 3)
            
            elif direction == 'diagonal':
                pos[0] += velocity
                pos[1] += velocity
                if pos[0] <= 1 or pos[0] >= self.grid_size - 2:
                    obs['velocity'] *= -1
                if pos[1] <= 1 or pos[1] >= self.grid_size - 2:
                    obs['velocity'] *= -1
                pos[0] = np.clip(pos[0], 2, self.grid_size - 3)
                pos[1] = np.clip(pos[1], 2, self.grid_size - 3)
    
    def _update_dynamic_obstacles(self):
        """Producer-consumer model: add/remove obstacles randomly"""
        if not self.enable_producer_consumer:
            return
        
        self.producer_consumer_timer += 1
        
        # Producer: Add new obstacle every 20 steps
        if self.producer_consumer_timer % 20 == 0 and len(self.dynamic_obstacles) < 5:
            pos = np.random.randint(3, self.grid_size-3, size=2)
            if (not np.array_equal(pos, self.robot_pos) and
                not np.array_equal(pos, self.goal_pos)):
                self.dynamic_obstacles.append({
                    'pos': pos,
                    'lifetime': random.randint(30, 60)  # Lives for 30-60 steps
                })
        
        # Consumer: Remove obstacles after lifetime
        self.dynamic_obstacles = [
            obs for obs in self.dynamic_obstacles
            if obs['lifetime'] > 0
        ]
        
        # Decrease lifetime
        for obs in self.dynamic_obstacles:
            obs['lifetime'] -= 1
    
    def _get_local_obstacle_map(self):
        """Get 3x3 grid around robot showing obstacles"""
        local_map = []
        
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                check_pos = self.robot_pos + np.array([dx, dy])
                
                # Check if obstacle at this position
                is_obstacle = False
                
                # Check static obstacles
                if any(np.array_equal(check_pos, obs) for obs in self.static_obstacles):
                    is_obstacle = True
                
                # Check moving obstacles
                if any(np.array_equal(check_pos, obs['pos']) for obs in self.moving_obstacles):
                    is_obstacle = True
                
                # Check dynamic obstacles
                if any(np.array_equal(check_pos, obs['pos']) for obs in self.dynamic_obstacles):
                    is_obstacle = True
                
                # Add position and obstacle flag
                if is_obstacle:
                    local_map.extend([check_pos[0], check_pos[1]])
                else:
                    local_map.extend([-1, -1])  # No obstacle
        
        return local_map
    
    def _get_obs(self):
        """Return observation with local obstacle information"""
        local_map = self._get_local_obstacle_map()
        
        obs = np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.goal_pos[0],
            self.goal_pos[1],
            *local_map
        ], dtype=np.float32)
        
        return obs
    
    def _get_info(self):
        """Return info"""
        distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        return {
            "distance_to_goal": distance,
            "steps": self.current_step,
            "is_success": np.array_equal(self.robot_pos, self.goal_pos),
            "moving_obstacles": len(self.moving_obstacles),
            "dynamic_obstacles": len(self.dynamic_obstacles)
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)
        
        self.robot_pos = np.array([1, 1])
        self._generate_obstacles()
        self.current_step = 0
        self.producer_consumer_timer = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one step with dynamic obstacle updates"""
        self.current_step += 1
        
        # Update environment dynamics BEFORE robot moves
        self._update_moving_obstacles()
        self._update_dynamic_obstacles()
        
        old_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        # Move robot
        new_pos = self.robot_pos.copy()
        if action == 0:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Down
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 2:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # Right
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        
        # Check collisions with ALL obstacle types
        collision = False
        
        # Static obstacles
        if any(np.array_equal(new_pos, obs) for obs in self.static_obstacles):
            collision = True
        
        # Moving obstacles
        if any(np.array_equal(new_pos, obs['pos']) for obs in self.moving_obstacles):
            collision = True
        
        # Dynamic obstacles
        if any(np.array_equal(new_pos, obs['pos']) for obs in self.dynamic_obstacles):
            collision = True
        
        # Calculate reward
        reward = 0
        terminated = False
        
        if collision:
            reward = -5
        else:
            self.robot_pos = new_pos
            new_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
            
            if np.array_equal(self.robot_pos, self.goal_pos):
                reward = 100
                terminated = True
            else:
                # Reward for getting closer
                distance_improvement = old_distance - new_distance
                reward = distance_improvement * 10 - 0.1
        
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render environment"""
        if self.render_mode == "rgb_array":
            img_size = 500
            cell_size = img_size // self.grid_size
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            # Draw grid
            for i in range(self.grid_size + 1):
                pos = i * cell_size
                img[pos, :] = [200, 200, 200]
                img[:, pos] = [200, 200, 200]
            
            # Draw static obstacles (dark red)
            for obs in self.static_obstacles:
                x, y = obs[0] * cell_size, obs[1] * cell_size
                img[y+2:y+cell_size-2, x+2:x+cell_size-2] = [200, 0, 0]
            
            # Draw moving obstacles (orange)
            for obs in self.moving_obstacles:
                pos = obs['pos']
                x, y = pos[0] * cell_size, pos[1] * cell_size
                img[y+2:y+cell_size-2, x+2:x+cell_size-2] = [255, 165, 0]
            
            # Draw dynamic obstacles (yellow - producer/consumer)
            for obs in self.dynamic_obstacles:
                pos = obs['pos']
                x, y = pos[0] * cell_size, pos[1] * cell_size
                img[y+2:y+cell_size-2, x+2:x+cell_size-2] = [255, 255, 0]
            
            # Draw goal (green)
            x, y = self.goal_pos[0] * cell_size, self.goal_pos[1] * cell_size
            img[y+2:y+cell_size-2, x+2:x+cell_size-2] = [0, 255, 0]
            
            # Draw robot (blue)
            x, y = self.robot_pos[0] * cell_size + cell_size//2, self.robot_pos[1] * cell_size + cell_size//2
            radius = cell_size // 3
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    if i*i + j*j <= radius*radius:
                        if 0 <= y+i < img_size and 0 <= x+j < img_size:
                            img[y+i, x+j] = [0, 0, 255]
            
            return img
        return None


class DynamicTrainingCallback(BaseCallback):
    """Callback for dynamic environment training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.success_count = 0
        self.episode_count = 0
        
    def _on_step(self):
        if len(self.locals.get('dones', [])) > 0:
            if self.locals['dones'][0]:
                self.episode_count += 1
                info = self.locals['infos'][0]
                
                if info.get('is_success', False):
                    self.success_count += 1
        
        return True


def train_dynamic_agent(algorithm='DQN', total_timesteps=150000):
    """Train agent in dynamic environment"""
    print("🌪️  Training in DYNAMIC Environment...")
    print("   Features: Moving obstacles + Producer-Consumer model")
    print("=" * 60)
    
    env = DynamicRobotEnv(
        grid_size=15,
        num_static_obstacles=8,
        num_moving_obstacles=5,
        enable_producer_consumer=True
    )
    
    callback = DynamicTrainingCallback()
    
    if algorithm == 'DQN':
        model = DQN(
            "MlpPolicy",
            env,
            learning_rate=0.0005,
            buffer_size=100000,
            learning_starts=2000,
            batch_size=64,
            gamma=0.99,
            exploration_fraction=0.4,
            exploration_final_eps=0.05,
            verbose=1
        )
    else:  # PPO
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=0.0005,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )
    
    print(f"\n📚 Training for {total_timesteps} timesteps (5-10 minutes)...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    model_name = f"dynamic_robot_{algorithm.lower()}"
    model.save(model_name)
    print(f"✅ Dynamic model saved as {model_name}.zip")
    
    if callback.episode_count > 0:
        success_rate = (callback.success_count / callback.episode_count) * 100
        print(f"🎯 Training Success Rate: {success_rate:.1f}%")
    
    return model, env


def create_dynamic_demo(model, num_episodes=3):
    """Create demo GIF showing dynamic obstacles"""
    print("\n🎬 Creating dynamic environment demo...")
    
    env = DynamicRobotEnv(
        grid_size=15,
        num_static_obstacles=8,
        num_moving_obstacles=5,
        enable_producer_consumer=True,
        render_mode='rgb_array'
    )
    
    all_frames = []
    
    for episode in range(num_episodes):
        print(f"   Recording episode {episode + 1}/{num_episodes}...")
        obs, info = env.reset()
        frames = []
        done = False
        
        while not done and len(frames) < 150:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            frame = env.render()
            if frame is not None:
                # Add label
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)
                
                # Add episode info
                text = f"Episode {episode+1} | Step {len(frames)} | Dynamic Env"
                try:
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
                except:
                    font = ImageFont.load_default()
                
                draw.text((10, 10), text, fill=(255, 255, 255), font=font)
                draw.text((10, 35), f"Moving: {info['moving_obstacles']} | Dynamic: {info['dynamic_obstacles']}", 
                         fill=(255, 255, 255), font=font)
                
                frames.append(np.array(img))
        
        all_frames.extend(frames)
        
        # Add pause between episodes
        if episode < num_episodes - 1:
            all_frames.extend([frames[-1]] * 10)
    
    # Save GIF
    images = [Image.fromarray(f) for f in all_frames]
    images[0].save(
        'dynamic_environment_demo.gif',
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )
    
    print("✅ Dynamic demo saved as: dynamic_environment_demo.gif")
    env.close()


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🌪️  DYNAMIC ENVIRONMENT Path Planning")
    print("   Moving Obstacles + Producer-Consumer Model")
    print("=" * 60)
    
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Demo existing model
        model_file = sys.argv[2] if len(sys.argv) > 2 else "dynamic_robot_dqn.zip"
        
        if 'dqn' in model_file.lower():
            model = DQN.load(model_file)
        else:
            model = PPO.load(model_file)
        
        create_dynamic_demo(model, num_episodes=3)
    else:
        # Training mode
        print("\nSelect algorithm:")
        print("   1. DQN (Recommended)")
        print("   2. PPO")
        
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        algorithm = 'PPO' if choice == '2' else 'DQN'
        
        # Train
        model, env = train_dynamic_agent(algorithm=algorithm, total_timesteps=150000)
        
        # Create demo
        print("\n" + "=" * 60)
        create_dynamic_demo(model, num_episodes=3)
        
        env.close()
        
        print("\n✅ Done! Created:")
        print(f"   - Model: dynamic_robot_{algorithm.lower()}.zip")
        print("   - Demo: dynamic_environment_demo.gif")
        print("\nThis shows:")
        print("   🔴 Dark Red = Static obstacles")
        print("   🟠 Orange = Moving obstacles")
        print("   🟡 Yellow = Dynamic (producer-consumer) obstacles")
        print("   🔵 Blue = Robot")
        print("   🟢 Green = Goal")
