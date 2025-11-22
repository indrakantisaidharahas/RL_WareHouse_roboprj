"""
Robot Path Planning using Reinforcement Learning
Using Stable-Baselines3 + Gymnasium (Modern RL Stack)

IMPROVED VERSION: Better rewards, easier environment, no pygame dependency for headless

Installation:
pip install gymnasium stable-baselines3 numpy matplotlib pillow

Run this script: python robot_pathplanning.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

class RobotPathPlanningEnv(gym.Env):
    """
    Custom Gymnasium Environment for Robot Path Planning
    
    IMPROVED: Better reward shaping, easier environment
    """
    
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=15, num_obstacles=15, render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Simpler observation: just positions
        self.observation_space = spaces.Box(
            low=0, 
            high=grid_size-1, 
            shape=(4,),  # [robot_x, robot_y, goal_x, goal_y]
            dtype=np.float32
        )
        
        # Initialize environment
        self.robot_pos = np.array([1, 1])
        self.goal_pos = np.array([grid_size-2, grid_size-2])
        self.obstacles = self._generate_obstacles()
        self.max_steps = 150
        self.current_step = 0
        self.previous_distance = None
        
    def _generate_obstacles(self):
        """Generate random obstacles with guaranteed path"""
        obstacles = []
        while len(obstacles) < self.num_obstacles:
            pos = np.random.randint(2, self.grid_size-2, size=2)
            # Don't place obstacles on start, goal, or their immediate neighbors
            if (not np.array_equal(pos, [1, 1]) and 
                not np.array_equal(pos, self.goal_pos) and
                not (abs(pos[0] - 1) <= 1 and abs(pos[1] - 1) <= 1) and
                not (abs(pos[0] - self.goal_pos[0]) <= 1 and abs(pos[1] - self.goal_pos[1]) <= 1)):
                if not any(np.array_equal(pos, obs) for obs in obstacles):
                    obstacles.append(pos)
        return obstacles
    
    def _get_obs(self):
        """Return simplified observation"""
        return np.array([
            self.robot_pos[0],
            self.robot_pos[1],
            self.goal_pos[0],
            self.goal_pos[1]
        ], dtype=np.float32)
    
    def _get_info(self):
        """Return additional info"""
        distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        return {
            "distance_to_goal": distance, 
            "steps": self.current_step,
            "is_success": np.array_equal(self.robot_pos, self.goal_pos)
        }
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.robot_pos = np.array([1, 1])
        self.obstacles = self._generate_obstacles()
        self.current_step = 0
        self.previous_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one step in the environment - IMPROVED REWARDS"""
        self.current_step += 1
        old_pos = self.robot_pos.copy()
        old_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        
        # Move robot based on action
        new_pos = self.robot_pos.copy()
        if action == 0:  # Up
            new_pos[1] = max(0, new_pos[1] - 1)
        elif action == 1:  # Down
            new_pos[1] = min(self.grid_size - 1, new_pos[1] + 1)
        elif action == 2:  # Left
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 3:  # Right
            new_pos[0] = min(self.grid_size - 1, new_pos[0] + 1)
        
        # Check for collision with obstacles
        collision = any(np.array_equal(new_pos, obs) for obs in self.obstacles)
        
        # IMPROVED REWARD SHAPING
        reward = 0
        terminated = False
        
        if collision:
            reward = -5  # Smaller collision penalty
            # Don't move if collision
        else:
            # Move the robot
            self.robot_pos = new_pos
            new_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
            
            # Big reward for reaching goal
            if np.array_equal(self.robot_pos, self.goal_pos):
                reward = 100
                terminated = True
            else:
                # Dense reward shaping - reward getting closer
                distance_improvement = old_distance - new_distance
                reward = distance_improvement * 10  # Scale up the reward
                
                # Small penalty for each step to encourage efficiency
                reward -= 0.1
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render as RGB array (no pygame needed)"""
        if self.render_mode == "rgb_array":
            # Create image array
            img_size = 500
            cell_size = img_size // self.grid_size
            img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
            
            # Draw grid
            for i in range(self.grid_size + 1):
                pos = i * cell_size
                img[pos, :] = [200, 200, 200]
                img[:, pos] = [200, 200, 200]
            
            # Draw obstacles (red)
            for obs in self.obstacles:
                x, y = obs[0] * cell_size, obs[1] * cell_size
                img[y+2:y+cell_size-2, x+2:x+cell_size-2] = [255, 0, 0]
            
            # Draw goal (green)
            x, y = self.goal_pos[0] * cell_size, self.goal_pos[1] * cell_size
            img[y+2:y+cell_size-2, x+2:x+cell_size-2] = [0, 255, 0]
            
            # Draw robot (blue circle approximation)
            x, y = self.robot_pos[0] * cell_size + cell_size//2, self.robot_pos[1] * cell_size + cell_size//2
            radius = cell_size // 3
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    if i*i + j*j <= radius*radius:
                        if 0 <= y+i < img_size and 0 <= x+j < img_size:
                            img[y+i, x+j] = [0, 0, 255]
            
            return img
        return None


class TrainingCallback(BaseCallback):
    """Callback for logging training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.success_count = 0
        self.episode_count = 0
        
    def _on_step(self):
        # Check if episode ended
        if len(self.locals.get('dones', [])) > 0:
            if self.locals['dones'][0]:
                self.episode_count += 1
                info = self.locals['infos'][0]
                
                # Track success
                if info.get('is_success', False):
                    self.success_count += 1
                
                # Calculate reward
                if hasattr(self.locals, 'rewards'):
                    self.episode_rewards.append(sum(self.locals.get('rewards', [0])))
        
        return True


def train_agent(algorithm='DQN', total_timesteps=100000):
    """
    Train the RL agent with IMPROVED settings
    """
    print(f"🤖 Training {algorithm} agent...")
    print(f"   Grid: 15x15, Obstacles: 15, Steps: {total_timesteps}")
    
    # Create environment
    env = RobotPathPlanningEnv(grid_size=15, num_obstacles=15)
    
    # Create callback
    callback = TrainingCallback()
    
    # Create agent with BETTER hyperparameters
    if algorithm == 'DQN':
        model = DQN(
            "MlpPolicy", 
            env, 
            learning_rate=0.0005,  # Increased learning rate
            buffer_size=50000,
            learning_starts=1000,
            batch_size=64,
            tau=1.0,
            gamma=0.99,
            exploration_fraction=0.3,  # Less exploration time
            exploration_final_eps=0.05,  # More exploration at end
            target_update_interval=1000,
            verbose=1
        )
    elif algorithm == 'PPO':
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=0.0005,
            n_steps=2048,
            batch_size=128,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train
    print(f"📚 Training... (this will take ~2-5 minutes)")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    # Save model
    model_name = f"robot_pathplanning_{algorithm.lower()}"
    model.save(model_name)
    print(f"✅ Model saved as {model_name}.zip")
    
    # Calculate success rate
    if callback.episode_count > 0:
        success_rate = (callback.success_count / callback.episode_count) * 100
        print(f"🎯 Training Success Rate: {success_rate:.1f}% ({callback.success_count}/{callback.episode_count} episodes)")
    
    # Plot training progress
    if len(callback.episode_rewards) > 0:
        plot_training_progress(callback.episode_rewards, algorithm)
    
    return model, env


def plot_training_progress(rewards, algorithm):
    """Plot training rewards over episodes"""
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: All rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6, linewidth=0.5)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{algorithm} Training Progress - All Episodes')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    window_size = min(50, len(rewards) // 10)
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                label=f'{window_size}-episode moving average', linewidth=2, color='red')
        plt.legend()
    
    # Subplot 2: Recent performance
    plt.subplot(1, 2, 2)
    recent = rewards[-min(100, len(rewards)):]
    plt.plot(recent, linewidth=1)
    plt.xlabel('Recent Episodes')
    plt.ylabel('Total Reward')
    plt.title(f'Last {len(recent)} Episodes')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero reward')
    plt.legend()
    
    plt.tight_layout()
    filename = f'training_progress_{algorithm.lower()}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"📊 Training plot saved as {filename}")
    plt.close()  # Don't show, just save


def evaluate_agent(model, env, num_episodes=10, save_gif=False):
    """
    Evaluate trained agent
    """
    print(f"\n🎯 Evaluating agent for {num_episodes} episodes...")
    
    successes = 0
    total_rewards = []
    episode_lengths = []
    frames = [] if save_gif else None
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        episode_frames = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            # Save frame for first episode
            if save_gif and episode == 0:
                frame = env.render()
                if frame is not None:
                    episode_frames.append(frame)
        
        if terminated and info.get('is_success', False):
            successes += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        status = "✅ SUCCESS" if (terminated and info.get('is_success', False)) else "❌ FAILED"
        print(f"Episode {episode+1}: {status} | Reward = {episode_reward:.2f} | Steps = {steps}")
        
        if episode == 0 and save_gif:
            frames = episode_frames
    
    success_rate = (successes / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(episode_lengths)
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Success Rate: {success_rate:.1f}% ({successes}/{num_episodes})")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    
    # Save visualization
    if save_gif and frames:
        save_episode_gif(frames)
    
    return success_rate, avg_reward, avg_steps


def save_episode_gif(frames, filename='robot_demo.gif'):
    """Save episode as animated GIF"""
    if not frames:
        return
    
    print(f"\n🎬 Creating animation...")
    from PIL import Image
    
    images = [Image.fromarray(frame) for frame in frames]
    images[0].save(
        filename,
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )
    print(f"✅ Animation saved as {filename}")


def visualize_best_episode(model, env):
    """Create a static visualization of one episode"""
    print("\n🎨 Creating visualization...")
    
    obs, info = env.reset()
    frames = []
    done = False
    
    # Run one episode
    while not done and len(frames) < 200:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
    
    # Create figure with multiple frames
    if frames:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Robot Path Planning - Episode Progression', fontsize=16)
        
        # Show key frames
        indices = [0, len(frames)//5, 2*len(frames)//5, 3*len(frames)//5, 4*len(frames)//5, -1]
        
        for idx, (ax, frame_idx) in enumerate(zip(axes.flat, indices)):
            if frame_idx < len(frames):
                ax.imshow(frames[frame_idx])
                ax.set_title(f'Step {frame_idx}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('robot_episode_visualization.png', dpi=150, bbox_inches='tight')
        print("✅ Visualization saved as robot_episode_visualization.png")
        plt.close()


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🤖 Robot Path Planning with Reinforcement Learning")
    print("   Using Stable-Baselines3 + Gymnasium")
    print("   IMPROVED VERSION - Better rewards & easier environment")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == 'demo':
        # Demo mode
        model_file = sys.argv[2] if len(sys.argv) > 2 else "robot_pathplanning_dqn.zip"
        print(f"🎬 Loading model from {model_file}...")
        
        if 'dqn' in model_file.lower():
            model = DQN.load(model_file)
        elif 'ppo' in model_file.lower():
            model = PPO.load(model_file)
        else:
            print("❌ Cannot determine algorithm from filename")
            sys.exit(1)
        
        env = RobotPathPlanningEnv(grid_size=15, num_obstacles=15, render_mode='rgb_array')
        evaluate_agent(model, env, num_episodes=10, save_gif=True)
        visualize_best_episode(model, env)
        env.close()
        
    else:
        # Training mode
        print("\n📚 Select algorithm:")
        print("   1. DQN (Deep Q-Network) - Recommended for this task")
        print("   2. PPO (Proximal Policy Optimization) - More stable")
        
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        algorithm = 'PPO' if choice == '2' else 'DQN'
        
        # Train with more timesteps
        model, env = train_agent(algorithm=algorithm, total_timesteps=100000)
        
        # Evaluate
        print("\n" + "="*60)
        success_rate, _, _ = evaluate_agent(model, env, num_episodes=20, save_gif=False)
        
        # Create visualizations
        env_render = RobotPathPlanningEnv(grid_size=15, num_obstacles=15, render_mode='rgb_array')
        visualize_best_episode(model, env_render)
        
        # Offer to create GIF
        if success_rate > 20:
            print("\n🎬 Your robot learned successfully!")
            gif = input("Create animated GIF demo? (y/n, default=y): ").strip().lower()
            if gif != 'n':
                evaluate_agent(model, env_render, num_episodes=1, save_gif=True)
        else:
            print("\n⚠️  Low success rate. Consider:")
            print("   - Training longer (increase timesteps)")
            print("   - Adjusting reward function")
            print("   - Reducing obstacles")
        
        env_render.close()
        env.close()
        
        print("\n✅ Done! Files created:")
        print(f"   - Model: robot_pathplanning_{algorithm.lower()}.zip")
        print(f"   - Graph: training_progress_{algorithm.lower()}.png")
        print("   - Visualization: robot_episode_visualization.png")
        if success_rate > 20:
            print("   - Animation: robot_demo.gif")
