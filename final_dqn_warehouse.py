"""
Robot Path Planning in DYNAMIC Environments (DQN) - FIXED
- Single agent
- Pickup → Delivery task (like 1.py)
- Static + moving + producer-consumer obstacles
- Path rendering + box + goal

Install:
    pip install gymnasium stable-baselines3 numpy pillow

Train:
    python dynamic_environment_dqn_pickup_delivery.py

Demo (after training):
    python dynamic_environment_dqn_pickup_delivery.py demo dynamic_robot_dqn.zip
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
import random
import os
import sys


class DynamicRobotEnv(gym.Env):
    """
    Dynamic Environment with:
    - Static obstacles (dark red)
    - Moving obstacles (orange)
    - Producer-consumer dynamic obstacles (yellow)

    Task:
    - Phase 1: Go from start -> package (pickup)
    - Phase 2: Go from package -> goal (delivery)
    """
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 10}

    def __init__(
        self,
        grid_size=15,
        num_static_obstacles=6,  # REDUCED for easier learning
        num_moving_obstacles=3,   # REDUCED
        enable_producer_consumer=True,
        render_mode=None
    ):
        super().__init__()

        self.grid_size = grid_size
        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles
        self.enable_producer_consumer = enable_producer_consumer
        self.render_mode = render_mode

        # Action space
        self.action_space = spaces.Discrete(4)

        # SIMPLIFIED OBSERVATION SPACE for better DQN learning
        # robot(2) + package(2) + goal(2) + carrying(1) + distance_to_package(1) + distance_to_goal(1)
        # + obstacle flags (4 directions)
        obs_size = 2 + 2 + 2 + 1 + 1 + 1 + 4
        self.observation_space = spaces.Box(
            low=-1.0,
            high=float(grid_size),
            shape=(obs_size,),
            dtype=np.float32
        )

        # State variables
        self.robot_pos = np.array([1, 1], dtype=float)
        self.package_pos = np.array([grid_size // 2, grid_size // 2], dtype=float)
        self.goal_pos = np.array([grid_size - 2, grid_size - 2], dtype=float)
        self.carrying = False

        self.static_obstacles = []
        self.moving_obstacles = []
        self.dynamic_obstacles = []

        self.max_steps = 300  # INCREASED
        self.current_step = 0
        self.producer_consumer_timer = 0

        self.path = []

    def _generate_obstacles(self):
        """Generate static and moving obstacles"""
        # Static obstacles
        self.static_obstacles = []
        attempts = 0
        while len(self.static_obstacles) < self.num_static_obstacles and attempts < 100:
            pos = np.random.randint(3, self.grid_size - 3, size=2).astype(float)
            if (
                np.linalg.norm(pos - self.robot_pos) > 2 and
                np.linalg.norm(pos - self.goal_pos) > 2 and
                np.linalg.norm(pos - self.package_pos) > 2
            ):
                if not any(np.allclose(pos, obs, atol=0.5) for obs in self.static_obstacles):
                    self.static_obstacles.append(pos)
            attempts += 1

        # Moving obstacles
        self.moving_obstacles = []
        for _ in range(self.num_moving_obstacles):
            pos = np.random.randint(4, self.grid_size - 4, size=2).astype(float)
            direction = random.choice(['horizontal', 'vertical'])
            velocity = random.choice([-1, 1])
            self.moving_obstacles.append({
                'pos': pos,
                'direction': direction,
                'velocity': velocity,
            })

        self.dynamic_obstacles = []

    def _update_moving_obstacles(self):
        """Update positions of moving obstacles"""
        for obs in self.moving_obstacles:
            pos = obs['pos']
            direction = obs['direction']
            velocity = obs['velocity']

            if direction == 'horizontal':
                pos[0] += velocity
                if pos[0] <= 2 or pos[0] >= self.grid_size - 3:
                    obs['velocity'] *= -1
                    pos[0] = np.clip(pos[0], 3, self.grid_size - 4)

            elif direction == 'vertical':
                pos[1] += velocity
                if pos[1] <= 2 or pos[1] >= self.grid_size - 3:
                    obs['velocity'] *= -1
                    pos[1] = np.clip(pos[1], 3, self.grid_size - 4)

    def _update_dynamic_obstacles(self):
        """Producer-consumer model"""
        if not self.enable_producer_consumer:
            return

        self.producer_consumer_timer += 1

        # Producer: add every 30 steps (slower)
        if self.producer_consumer_timer % 30 == 0 and len(self.dynamic_obstacles) < 3:
            pos = np.random.randint(4, self.grid_size - 4, size=2).astype(float)
            if (
                np.linalg.norm(pos - self.robot_pos) > 2 and
                np.linalg.norm(pos - self.goal_pos) > 2
            ):
                self.dynamic_obstacles.append({
                    'pos': pos,
                    'lifetime': random.randint(40, 80)
                })

        # Consumer
        for obs in self.dynamic_obstacles:
            obs['lifetime'] -= 1
        self.dynamic_obstacles = [
            obs for obs in self.dynamic_obstacles if obs['lifetime'] > 0
        ]

    def _check_obstacle(self, pos):
        """Check if position has obstacle"""
        # Bounds
        if pos[0] < 0 or pos[0] >= self.grid_size or pos[1] < 0 or pos[1] >= self.grid_size:
            return True

        # Static
        for obs in self.static_obstacles:
            if np.allclose(pos, obs, atol=0.8):
                return True

        # Moving
        for obs in self.moving_obstacles:
            if np.allclose(pos, obs['pos'], atol=0.8):
                return True

        # Dynamic
        for obs in self.dynamic_obstacles:
            if np.allclose(pos, obs['pos'], atol=0.8):
                return True

        return False

    def _get_obstacle_flags(self):
        """Get obstacle flags in 4 directions (simplified)"""
        flags = []
        for delta in [[0, -1], [0, 1], [-1, 0], [1, 0]]:  # up, down, left, right
            test_pos = self.robot_pos + np.array(delta, dtype=float)
            flags.append(1.0 if self._check_obstacle(test_pos) else 0.0)
        return flags

    def _get_obs(self):
        """SIMPLIFIED observation for better DQN learning"""
        # Normalize positions
        robot_norm = self.robot_pos / self.grid_size
        package_norm = self.package_pos / self.grid_size
        goal_norm = self.goal_pos / self.grid_size

        # Distances
        dist_to_package = np.linalg.norm(self.robot_pos - self.package_pos) / self.grid_size
        dist_to_goal = np.linalg.norm(self.robot_pos - self.goal_pos) / self.grid_size

        # Obstacle flags
        obstacle_flags = self._get_obstacle_flags()

        obs = np.array([
            robot_norm[0],
            robot_norm[1],
            package_norm[0],
            package_norm[1],
            goal_norm[0],
            goal_norm[1],
            1.0 if self.carrying else 0.0,
            dist_to_package,
            dist_to_goal,
            *obstacle_flags
        ], dtype=np.float32)

        return obs

    def _get_info(self):
        """Return info dict"""
        if not self.carrying:
            distance = np.linalg.norm(self.robot_pos - self.package_pos)
        else:
            distance = np.linalg.norm(self.robot_pos - self.goal_pos)

        return {
            "distance_to_target": float(distance),
            "steps": self.current_step,
            "carrying": self.carrying,
            "is_success": self.carrying and np.allclose(self.robot_pos, self.goal_pos, atol=0.8),
            "moving_obstacles": len(self.moving_obstacles),
            "dynamic_obstacles": len(self.dynamic_obstacles),
        }

    def reset(self, seed=None, options=None):
        """Reset environment"""
        super().reset(seed=seed)

        self.robot_pos = np.array([1.0, 1.0], dtype=float)
        self.goal_pos = np.array([self.grid_size - 2.0, self.grid_size - 2.0], dtype=float)
        self.package_pos = np.array([self.grid_size / 2.0, self.grid_size / 2.0], dtype=float)
        self.carrying = False

        self._generate_obstacles()
        self.current_step = 0
        self.producer_consumer_timer = 0
        self.path = [self.robot_pos.copy()]

        return self._get_obs(), self._get_info()

    def step(self, action):
        """Execute one step - FIXED VERSION"""
        self.current_step += 1

        # Update environment
        self._update_moving_obstacles()
        self._update_dynamic_obstacles()

        # Store old position and distance
        old_pos = self.robot_pos.copy()
        
        # Calculate distance to CURRENT target
        if not self.carrying:
            target = self.package_pos
        else:
            target = self.goal_pos
        
        old_distance = np.linalg.norm(old_pos - target)

        # Move robot
        new_pos = self.robot_pos.copy()
        if action == 0:      # Up
            new_pos[1] -= 1
        elif action == 1:    # Down
            new_pos[1] += 1
        elif action == 2:    # Left
            new_pos[0] -= 1
        elif action == 3:    # Right
            new_pos[0] += 1

        # Clip to bounds
        new_pos[0] = np.clip(new_pos[0], 0, self.grid_size - 1)
        new_pos[1] = np.clip(new_pos[1], 0, self.grid_size - 1)

        # Check collision
        collision = self._check_obstacle(new_pos)

        reward = 0.0
        terminated = False

        if collision:
            # Don't move, penalty
            reward = -10.0
        else:
            # Update position
            self.robot_pos = new_pos
            
            # Recalculate distance after move
            new_distance = np.linalg.norm(self.robot_pos - target)

            # PHASE 1: Not carrying - go to package
            if not self.carrying:
                if np.allclose(self.robot_pos, self.package_pos, atol=1.0):
                    # PICKUP!
                    self.carrying = True
                    reward = 100.0
                    print(f"✅ Step {self.current_step}: PICKED UP package at {self.robot_pos}")
                else:
                    # Reward for getting closer to package
                    if new_distance < old_distance:
                        reward = 1.0
                    else:
                        reward = -0.5

            # PHASE 2: Carrying - go to goal
            else:
                if np.allclose(self.robot_pos, self.goal_pos, atol=1.0):
                    # DELIVERY!
                    reward = 200.0
                    terminated = True
                    print(f"🎉 Step {self.current_step}: DELIVERED at goal!")
                else:
                    # Reward for getting closer to goal
                    if new_distance < old_distance:
                        reward = 2.0  # Higher reward when carrying
                    else:
                        reward = -0.5

        # Small time penalty
        reward -= 0.1

        truncated = self.current_step >= self.max_steps
        self.path.append(self.robot_pos.copy())

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """Render environment"""
        if self.render_mode != "rgb_array":
            return None

        img_size = 600
        cell_size = img_size // self.grid_size
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255

        # Grid - FIXED: don't go to grid_size+1
        for i in range(self.grid_size):  # Changed from range(self.grid_size + 1)
            pos = i * cell_size
            if pos < img_size:  # Safety check
                img[pos, :] = [220, 220, 220]
                img[:, pos] = [220, 220, 220]

        # Static obstacles (dark red)
        for obs in self.static_obstacles:
            x, y = int(obs[0] * cell_size), int(obs[1] * cell_size)
            if 0 <= y < img_size and 0 <= x < img_size:
                y_end = min(y + cell_size - 2, img_size)
                x_end = min(x + cell_size - 2, img_size)
                img[y+2:y_end, x+2:x_end] = [180, 0, 0]

        # Moving obstacles (orange)
        for obs in self.moving_obstacles:
            pos = obs['pos']
            x, y = int(pos[0] * cell_size), int(pos[1] * cell_size)
            if 0 <= y < img_size and 0 <= x < img_size:
                y_end = min(y + cell_size - 2, img_size)
                x_end = min(x + cell_size - 2, img_size)
                img[y+2:y_end, x+2:x_end] = [255, 140, 0]

        # Dynamic obstacles (yellow)
        for obs in self.dynamic_obstacles:
            pos = obs['pos']
            x, y = int(pos[0] * cell_size), int(pos[1] * cell_size)
            if 0 <= y < img_size and 0 <= x < img_size:
                y_end = min(y + cell_size - 2, img_size)
                x_end = min(x + cell_size - 2, img_size)
                img[y+2:y_end, x+2:x_end] = [255, 220, 0]

        # Package (cyan) - only if not carrying
        if not self.carrying:
            px, py = int(self.package_pos[0] * cell_size), int(self.package_pos[1] * cell_size)
            if 0 <= py < img_size and 0 <= px < img_size:
                py_end = min(py + cell_size - 5, img_size)
                px_end = min(px + cell_size - 5, img_size)
                img[py+5:py_end, px+5:px_end] = [0, 255, 255]

        # Goal (green)
        gx, gy = int(self.goal_pos[0] * cell_size), int(self.goal_pos[1] * cell_size)
        if 0 <= gy < img_size and 0 <= gx < img_size:
            gy_end = min(gy + cell_size - 5, img_size)
            gx_end = min(gx + cell_size - 5, img_size)
            img[gy+5:gy_end, gx+5:gx_end] = [0, 200, 0]

        # Path (light purple)
        if len(self.path) > 1:
            for i in range(1, len(self.path)):
                p1 = self.path[i - 1]
                p2 = self.path[i]
                x1 = int(p1[0] * cell_size + cell_size // 2)
                y1 = int(p1[1] * cell_size + cell_size // 2)
                x2 = int(p2[0] * cell_size + cell_size // 2)
                y2 = int(p2[1] * cell_size + cell_size // 2)
                
                # Draw line
                num = max(abs(x2 - x1), abs(y2 - y1)) + 1
                for t in range(num):
                    xx = int(x1 + (x2 - x1) * t / num)
                    yy = int(y1 + (y2 - y1) * t / num)
                    if 0 <= yy < img_size and 0 <= xx < img_size:
                        img[yy, xx] = [200, 100, 255]

        # Robot (blue when empty, bright green when carrying)
        if self.carrying:
            color = [0, 255, 0]  # Bright green
        else:
            color = [0, 100, 255]  # Blue

        rx = int(self.robot_pos[0] * cell_size + cell_size // 2)
        ry = int(self.robot_pos[1] * cell_size + cell_size // 2)
        radius = cell_size // 3
        
        for i in range(-radius, radius + 1):
            for j in range(-radius, radius + 1):
                if i * i + j * j <= radius * radius:
                    yy = ry + i
                    xx = rx + j
                    if 0 <= yy < img_size and 0 <= xx < img_size:
                        img[yy, xx] = color

        return img



class DynamicTrainingCallback(BaseCallback):
    """Callback for training"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.success_count = 0
        self.pickup_count = 0

    def _on_step(self):
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if dones is not None and len(dones) > 0 and dones[0]:
            self.episode_count += 1
            if infos is not None and len(infos) > 0:
                info = infos[0]
                if info.get("carrying", False):
                    self.pickup_count += 1
                if info.get("is_success", False):
                    self.success_count += 1
                
                if self.episode_count % 50 == 0:
                    pickup_rate = (self.pickup_count / 50) * 100
                    success_rate = (self.success_count / 50) * 100
                    print(f"\nEpisodes {self.episode_count}: Pickup={pickup_rate:.1f}%, Delivery={success_rate:.1f}%")
                    self.pickup_count = 0
                    self.success_count = 0

        return True


def train_dynamic_agent(total_timesteps=200000):
    """Train DQN agent"""
    print("🌪️  Training DQN in DYNAMIC Environment (Pickup → Delivery)")
    print("=" * 70)

    env = DynamicRobotEnv(
        grid_size=15,
        num_static_obstacles=6,
        num_moving_obstacles=3,
        enable_producer_consumer=True
    )

    callback = DynamicTrainingCallback()

    # DQN with better hyperparameters
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0003,
        buffer_size=100000,
        learning_starts=5000,
        batch_size=128,
        gamma=0.99,
        exploration_fraction=0.3,  # Explore less
        exploration_initial_eps=1.0,
        exploration_final_eps=0.02,  # Lower final epsilon
        target_update_interval=1000,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256])  # Bigger network
    )

    print(f"\n📚 Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)

    model_name = "dynamic_robot_dqn"
    model.save(model_name)
    print(f"\n✅ Model saved as {model_name}.zip")

    env.close()
    return model


def create_dynamic_demo(model, num_episodes=5):
    """Create demo GIF"""
    print("\n🎬 Creating demo...")

    env = DynamicRobotEnv(
        grid_size=15,
        num_static_obstacles=6,
        num_moving_obstacles=3,
        enable_producer_consumer=True,
        render_mode='rgb_array'
    )

    all_frames = []

    for episode in range(num_episodes):
        print(f"   Episode {episode + 1}/{num_episodes}...")
        obs, info = env.reset()
        frames = []
        done = False
        step_count = 0

        while not done and step_count < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

            frame = env.render()
            if frame is not None:
                img = Image.fromarray(frame)
                draw = ImageDraw.Draw(img)

                try:
                    font = ImageFont.truetype("arial.ttf", 18)
                except:
                    font = ImageFont.load_default()

                status = "CARRYING" if info['carrying'] else "SEEKING"
                draw.text((10, 10), f"Ep {episode+1} | Step {step_count} | {status}", fill=(0, 0, 0), font=font)

                frames.append(np.array(img))

        all_frames.extend(frames)
        
        result = "✅ SUCCESS" if info.get('is_success') else ("📦 PICKUP" if info['carrying'] else "❌ FAILED")
        print(f"      {result} in {step_count} steps")

        if episode < num_episodes - 1 and len(frames) > 0:
            all_frames.extend([frames[-1]] * 15)

    if all_frames:
        images = [Image.fromarray(f) for f in all_frames]
        images[0].save(
            'dynamic_environment_demo.gif',
            save_all=True,
            append_images=images[1:],
            duration=80,
            loop=0
        )
        print("\n✅ Demo saved: dynamic_environment_demo.gif")

    env.close()


if __name__ == "__main__":
    print("=" * 70)
    print("🌪️  DYNAMIC ENVIRONMENT - DQN (FIXED)")
    print("=" * 70)

    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        model_file = sys.argv[2] if len(sys.argv) > 2 else "dynamic_robot_dqn.zip"
        if not os.path.exists(model_file):
            print(f"❌ Model not found: {model_file}")
            sys.exit(1)
        model = DQN.load(model_file)
        create_dynamic_demo(model, num_episodes=5)
    else:
        model = train_dynamic_agent(total_timesteps=200000)
        print("\n" + "=" * 70)
        create_dynamic_demo(model, num_episodes=5)
        print("\n✅ Complete!")
        print("\nLegend:")
        print("   🔴 Dark Red = Static obstacles")
        print("   🟠 Orange   = Moving obstacles")
        print("   🟡 Yellow   = Dynamic obstacles")
        print("   🔵 Blue     = Robot (seeking)")
        print("   🟢 Green    = Robot (carrying)")
        print("   🟦 Cyan     = Package")
        print("   🟩 Green sq = Goal")