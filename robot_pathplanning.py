"""
Robot Path Planning using Reinforcement Learning
Using Stable-Baselines3 + Gymnasium (Modern RL Stack)

Installation:
pip install gymnasium stable-baselines3 numpy matplotlib pygame

Run this script: python robot_pathplanning.py
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

class RobotPathPlanningEnv(gym.Env):
    """
    Custom Gymnasium Environment for Robot Path Planning
    
    Observation Space: Robot position (x, y) and goal position
    Action Space: 4 discrete actions (Up, Down, Left, Right)
    Reward: +100 for goal, -10 for obstacle, -1 per step, +5 for getting closer
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}
    
    def __init__(self, grid_size=20, num_obstacles=30, render_mode=None):
        super().__init__()
        
        self.grid_size = grid_size
        self.num_obstacles = num_obstacles
        self.render_mode = render_mode
        
        # Define action and observation space
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Observation: [robot_x, robot_y, goal_x, goal_y, obstacle_map_flattened]
        self.observation_space = spaces.Box(
            low=0, 
            high=grid_size-1, 
            shape=(4 + grid_size*grid_size,), 
            dtype=np.float32
        )
        
        # Initialize environment
        self.robot_pos = np.array([1, 1])
        self.goal_pos = np.array([grid_size-2, grid_size-2])
        self.obstacles = self._generate_obstacles()
        self.max_steps = 200
        self.current_step = 0
        
        # For rendering
        self.window = None
        self.clock = None
        
    def _generate_obstacles(self):
        """Generate random obstacles"""
        obstacles = []
        while len(obstacles) < self.num_obstacles:
            pos = np.random.randint(2, self.grid_size-2, size=2)
            # Don't place obstacles on start or goal
            if not np.array_equal(pos, [1, 1]) and not np.array_equal(pos, self.goal_pos):
                if not any(np.array_equal(pos, obs) for obs in obstacles):
                    obstacles.append(pos)
        return obstacles
    
    def _get_obs(self):
        """Return current observation"""
        obstacle_map = np.zeros((self.grid_size, self.grid_size))
        for obs in self.obstacles:
            obstacle_map[obs[0], obs[1]] = 1
        
        obs = np.concatenate([
            self.robot_pos.astype(np.float32),
            self.goal_pos.astype(np.float32),
            obstacle_map.flatten().astype(np.float32)
        ])
        return obs
    
    def _get_info(self):
        """Return additional info"""
        distance = np.linalg.norm(self.robot_pos - self.goal_pos)
        return {"distance_to_goal": distance, "steps": self.current_step}
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.robot_pos = np.array([1, 1])
        self.obstacles = self._generate_obstacles()
        self.current_step = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action):
        """Execute one step in the environment"""
        self.current_step += 1
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
        
        # Calculate reward
        reward = -1  # Step penalty
        terminated = False
        
        if collision:
            reward = -10  # Collision penalty
        else:
            self.robot_pos = new_pos
            new_distance = np.linalg.norm(self.robot_pos - self.goal_pos)
            
            # Reward for getting closer
            if new_distance < old_distance:
                reward += 5
            
            # Check if goal reached
            if np.array_equal(self.robot_pos, self.goal_pos):
                reward = 100
                terminated = True
        
        # Check if max steps reached
        truncated = self.current_step >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()
    
    def render(self):
        """Render the environment (optional)"""
        if self.render_mode == "human":
            import pygame
            
            if self.window is None:
                pygame.init()
                self.cell_size = 30
                window_size = self.grid_size * self.cell_size
                self.window = pygame.display.set_mode((window_size, window_size))
                pygame.display.set_caption("Robot Path Planning")
            
            if self.clock is None:
                self.clock = pygame.time.Clock()
            
            canvas = pygame.Surface((self.grid_size * self.cell_size, self.grid_size * self.cell_size))
            canvas.fill((255, 255, 255))
            
            # Draw grid
            for i in range(self.grid_size + 1):
                pygame.draw.line(canvas, (200, 200, 200), 
                               (0, i * self.cell_size), 
                               (self.grid_size * self.cell_size, i * self.cell_size))
                pygame.draw.line(canvas, (200, 200, 200), 
                               (i * self.cell_size, 0), 
                               (i * self.cell_size, self.grid_size * self.cell_size))
            
            # Draw obstacles
            for obs in self.obstacles:
                pygame.draw.rect(canvas, (255, 0, 0), 
                               (obs[0] * self.cell_size + 2, 
                                obs[1] * self.cell_size + 2, 
                                self.cell_size - 4, 
                                self.cell_size - 4))
            
            # Draw goal
            pygame.draw.rect(canvas, (0, 255, 0), 
                           (self.goal_pos[0] * self.cell_size + 2, 
                            self.goal_pos[1] * self.cell_size + 2, 
                            self.cell_size - 4, 
                            self.cell_size - 4))
            
            # Draw robot
            pygame.draw.circle(canvas, (0, 0, 255), 
                             (self.robot_pos[0] * self.cell_size + self.cell_size // 2,
                              self.robot_pos[1] * self.cell_size + self.cell_size // 2),
                             self.cell_size // 3)
            
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


class TrainingCallback(BaseCallback):
    """
    Callback for logging training progress
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self):
        if self.locals.get('dones'):
            if self.locals['dones'][0]:
                self.episode_rewards.append(self.locals['rewards'][0])
                self.episode_lengths.append(self.locals['infos'][0].get('steps', 0))
        return True


def train_agent(algorithm='DQN', total_timesteps=50000):
    """
    Train the RL agent
    
    Args:
        algorithm: 'DQN' or 'PPO'
        total_timesteps: Number of training steps
    """
    print(f"🤖 Training {algorithm} agent...")
    
    # Create environment
    env = RobotPathPlanningEnv(grid_size=20, num_obstacles=30)
    
    # Create callback
    callback = TrainingCallback()
    
    # Create agent
    if algorithm == 'DQN':
        model = DQN(
            "MlpPolicy", 
            env, 
            learning_rate=0.0001,
            buffer_size=10000,
            learning_starts=1000,
            batch_size=32,
            tau=1.0,
            gamma=0.99,
            exploration_fraction=0.5,
            exploration_final_eps=0.01,
            verbose=1
        )
    elif algorithm == 'PPO':
        model = PPO(
            "MlpPolicy", 
            env, 
            learning_rate=0.0003,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            verbose=1
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Train
    print(f"Training for {total_timesteps} timesteps...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    # Save model
    model_name = f"robot_pathplanning_{algorithm.lower()}"
    model.save(model_name)
    print(f"✅ Model saved as {model_name}.zip")
    
    # Plot training progress
    plot_training_progress(callback.episode_rewards, algorithm)
    
    return model, env


def plot_training_progress(rewards, algorithm):
    """Plot training rewards over episodes"""
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title(f'{algorithm} Training Progress')
    plt.grid(True, alpha=0.3)
    
    # Add moving average
    window_size = 50
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards)), moving_avg, 
                label=f'{window_size}-episode moving average', linewidth=2)
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'training_progress_{algorithm.lower()}.png', dpi=150)
    print(f"📊 Training plot saved as training_progress_{algorithm.lower()}.png")
    plt.show()


def evaluate_agent(model, env, num_episodes=10, render=False):
    """
    Evaluate trained agent
    
    Args:
        model: Trained model
        env: Environment
        num_episodes: Number of episodes to evaluate
        render: Whether to render the episodes
    """
    print(f"\n🎯 Evaluating agent for {num_episodes} episodes...")
    
    successes = 0
    total_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        steps = 0
        done = False
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if render:
                env.render()
        
        if terminated:  # Reached goal
            successes += 1
        
        total_rewards.append(episode_reward)
        episode_lengths.append(steps)
        
        print(f"Episode {episode+1}: Reward = {episode_reward:.2f}, Steps = {steps}, "
              f"Success = {terminated}")
    
    success_rate = (successes / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    avg_steps = np.mean(episode_lengths)
    
    print(f"\n📈 Evaluation Results:")
    print(f"   Success Rate: {success_rate:.1f}%")
    print(f"   Average Reward: {avg_reward:.2f}")
    print(f"   Average Steps: {avg_steps:.1f}")
    
    return success_rate, avg_reward, avg_steps


def demo_trained_agent(model_path, num_episodes=5):
    """
    Load and demonstrate a trained agent
    
    Args:
        model_path: Path to saved model
        num_episodes: Number of episodes to demonstrate
    """
    print(f"🎬 Loading model from {model_path}...")
    
    # Determine algorithm from filename
    if 'dqn' in model_path.lower():
        model = DQN.load(model_path)
    elif 'ppo' in model_path.lower():
        model = PPO.load(model_path)
    else:
        raise ValueError("Cannot determine algorithm from filename")
    
    # Create environment with rendering
    env = RobotPathPlanningEnv(grid_size=20, num_obstacles=30, render_mode='human')
    
    # Evaluate
    evaluate_agent(model, env, num_episodes=num_episodes, render=True)
    
    env.close()


if __name__ == "__main__":
    import sys
    
    print("=" * 60)
    print("🤖 Robot Path Planning with Reinforcement Learning")
    print("   Using Stable-Baselines3 + Gymnasium")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            # Demo mode
            model_file = sys.argv[2] if len(sys.argv) > 2 else "robot_pathplanning_dqn.zip"
            demo_trained_agent(model_file)
        else:
            print("Usage: python robot_pathplanning.py [demo <model_file>]")
    else:
        # Training mode
        print("\n📚 Select algorithm:")
        print("   1. DQN (Deep Q-Network) - Good for discrete action spaces")
        print("   2. PPO (Proximal Policy Optimization) - More stable training")
        
        choice = input("\nEnter choice (1 or 2, default=1): ").strip()
        algorithm = 'PPO' if choice == '2' else 'DQN'
        
        # Train
        model, env = train_agent(algorithm=algorithm, total_timesteps=50000)
        
        # Evaluate
        evaluate_agent(model, env, num_episodes=10, render=False)
        
        # Ask if user wants to see demo
        demo = input("\n🎬 Show visual demo? (y/n, default=y): ").strip().lower()
        if demo != 'n':
            env_render = RobotPathPlanningEnv(grid_size=20, num_obstacles=30, render_mode='human')
            evaluate_agent(model, env_render, num_episodes=5, render=True)
            env_render.close()
        
        env.close()
        
        print("\n✅ Done! Model saved and ready to use.")
        print(f"   To demo later: python robot_pathplanning.py demo robot_pathplanning_{algorithm.lower()}.zip")
