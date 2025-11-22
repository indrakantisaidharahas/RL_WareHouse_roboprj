# turtlebot3_rl_controller.py
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os

# --- Webots Import ---
WEBOTS_LIB_PATH = '/usr/local/webots/lib/controller/python'
if WEBOTS_LIB_PATH not in sys.path:
    sys.path.append(WEBOTS_LIB_PATH)

try:
    from controller import Supervisor
except ImportError as e:
    print(f"FATAL ERROR: Failed to import Supervisor: {e}")
    sys.exit(1)

# --- RL Libraries Import ---
try:
    from stable_baselines3 import PPO
except ImportError:
    print('FATAL ERROR: Stable-Baselines3 or Gymnasium not installed.')
    sys.exit(1)


class TurtleBotGymEnvironment(Supervisor, gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, max_episode_steps=1000, render_mode=None):
        print("DEBUG: Entering TurtleBotGymEnvironment.__init__")
        try:
            Supervisor.__init__(self)
        except Exception as e:
            print(f"FATAL ERROR during Supervisor.__init__(): {e}")
            sys.exit(1)

        # --- Basic Setup ---
        self.timestep = int(self.getBasicTimeStep())
        self.max_steps = max_episode_steps
        self.steps_taken = 0
        self.render_mode = render_mode
        self.devices_initialized = False
        
        # Device handles
        self.robot_node = None
        self.left_motor = None
        self.right_motor = None
        self.lidar = None
        
        # TurtleBot3 parameters
        self.max_motor_velocity = 2.84  # TurtleBot3 max speed (rad/s)
        self.wheel_radius = 0.033  # meters
        self.wheel_separation = 0.160  # meters between wheels
        self.lidar_points = 360
        self.lidar_max_range = 3.5

        print(f"DEBUG: Timestep set to {self.timestep}")

        # --- Define Spaces ---
        self.action_space = spaces.Box(
            low=-self.max_motor_velocity, 
            high=self.max_motor_velocity, 
            shape=(2,), 
            dtype=np.float32
        )
        
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.lidar_points,), 
            dtype=np.float32
        )

        self.spec = gym.envs.registration.EnvSpec(
            id='WebotsTurtleBotEnv-v0', 
            max_episode_steps=max_episode_steps
        )
        print("DEBUG: __init__ finished successfully.")

    def _webots_step(self):
        """Wrapper for Webots simulation step"""
        return Supervisor.step(self, self.timestep)

    def _setup_devices(self):
        """Get handles to devices - called only once in reset."""
        print("DEBUG: _setup_devices() called.")
        
        # Get the robot node
        self.robot_node = self.getFromDef("TURTLEBOT")
        if self.robot_node is None:
            print("WARNING: Robot DEF 'TURTLEBOT' not found. Using self.")
            self.robot_node = self.getSelf()
        else:
            print(f"DEBUG: Found robot node with DEF 'TURTLEBOT'")
        
        if self.robot_node is None:
            print("ERROR: Could not get robot node!")
            return False

        # List all available devices for debugging
        print("DEBUG: Listing all available devices...")
        num_devices = self.getNumberOfDevices()
        device_list = []
        for i in range(num_devices):
            try:
                device = self.getDeviceByIndex(i)
                if device:
                    device_list.append(device.getName())
            except:
                pass
        print(f"DEBUG: Found {len(device_list)} devices: {device_list}")
        
        # Get TurtleBot3 devices - try multiple names
        # TurtleBot3 Burger motor names
        motor_names = ["left wheel motor", "right wheel motor", 
                       "wheel_left_joint", "wheel_right_joint",
                       "left_wheel_motor", "right_wheel_motor"]
        
        for name in motor_names:
            if self.left_motor is None:
                self.left_motor = self.getDevice(name)
                if self.left_motor:
                    print(f"DEBUG: Found left motor as '{name}'")
            if "left" not in name and self.right_motor is None:
                self.right_motor = self.getDevice(name)
                if self.right_motor:
                    print(f"DEBUG: Found right motor as '{name}'")
        
        # Get lidar - try multiple names
        lidar_names = ["LDS-01", "lidar", "LiDAR", "lds", "laser"]
        for name in lidar_names:
            self.lidar = self.getDevice(name)
            if self.lidar:
                print(f"DEBUG: Found lidar as '{name}'")
                break

        if self.left_motor is None or self.right_motor is None:
            print("ERROR: Could not find motor devices!")
            print(f"Available devices: {device_list}")
            return False
            
        if self.lidar is None:
            print("ERROR: Could not find lidar!")
            print(f"Available devices: {device_list}")
            return False

        # Enable sensor and configure motors
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()
        
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Step a few times to let sensor initialize
        for i in range(5):
            if self._webots_step() == -1:
                print(f"WARNING: Sim step {i} returned -1 during setup")
                continue

        # Update parameters with actual values
        try:
            self.max_motor_velocity = self.left_motor.getMaxVelocity()
        except:
            pass
        
        self.lidar_points = self.lidar.getNumberOfPoints()
        self.lidar_max_range = self.lidar.getMaxRange()

        if self.lidar_points <= 0:
            print("ERROR: Lidar points invalid after setup!")
            return False

        self.action_space = spaces.Box(
            low=-self.max_motor_velocity, 
            high=self.max_motor_velocity, 
            shape=(2,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(self.lidar_points,), 
            dtype=np.float32
        )

        print(f"DEBUG: Devices setup complete.")
        print(f"DEBUG: Max motor velocity: {self.max_motor_velocity}")
        print(f"DEBUG: Lidar points: {self.lidar_points}")
        print(f"DEBUG: Lidar max range: {self.lidar_max_range}")
        
        self.devices_initialized = True
        return True

    def _get_obs(self):
        """Get observation from lidar."""
        if not self.devices_initialized or self.lidar is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        lidar_values = self.lidar.getRangeImage()
        
        if lidar_values is None or len(lidar_values) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        # Convert to numpy array
        lidar_values = np.array(lidar_values, dtype=np.float32)
        
        # Filter out readings below threshold (robot's own body)
        lidar_values[lidar_values < 0.1] = self.lidar_max_range
        
        # Normalize to 0-1 range
        lidar_values = lidar_values / self.lidar_max_range
        lidar_values = np.clip(lidar_values, 0.0, 1.0)
        
        # Handle inf values
        lidar_values[~np.isfinite(lidar_values)] = 1.0

        if len(lidar_values) != self.lidar_points:
            print(f"WARNING: Expected {self.lidar_points} lidar points, got {len(lidar_values)}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        return lidar_values

    def reset(self, seed=None, options=None):
        """Reset simulation and return initial observation."""
        super().reset(seed=seed)

        # Setup devices on first reset
        if not self.devices_initialized:
            print("DEBUG: First reset - setting up devices...")
            if not self._setup_devices():
                print("ERROR: Device setup failed!")
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        # Reset internals first
        self.steps_taken = 0
        
        # Stop motors
        try:
            self.left_motor.setVelocity(0.0)
            self.right_motor.setVelocity(0.0)
        except:
            pass

        # Reset simulation
        self.simulationReset()
        
        # Wait for reset to complete
        for _ in range(3):
            if self._webots_step() == -1:
                break
        
        # Reposition the robot
        if self.robot_node:
            translation_field = self.robot_node.getField("translation")
            rotation_field = self.robot_node.getField("rotation")
            
            if translation_field and rotation_field:
                try:
                    # Arena is 5x5 meters, keep robot safely inside
                    arena_size = 2.0
                    new_x = np.random.uniform(-arena_size, arena_size)
                    new_z = np.random.uniform(-arena_size, arena_size)
                    new_rot = np.random.uniform(-np.pi, np.pi)
                    
                    # TurtleBot3 origin is at the center, Y should be at wheel height
                    translation_field.setSFVec3f([new_x, 0.0, new_z])
                    rotation_field.setSFRotation([0, 1, 0, new_rot])
                except Exception as e:
                    print(f"WARNING: Could not reposition: {e}")

        # Step a few times for stabilization
        for _ in range(5):
            if self._webots_step() == -1:
                break

        # Get initial observation
        obs = self._get_obs()
        
        # Set target position for this episode
        self.target_position = [2.0, 0.0, 2.0]  # Goal location
        
        info = {}
        return obs, info

    def step(self, action):
        """Apply action, step simulation, calculate reward, check done."""
        if not self.devices_initialized:
            print("ERROR: Devices not initialized!")
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, True, {}

        self.steps_taken += 1

        # Apply action
        try:
            left_speed = np.clip(float(action[0]), -self.max_motor_velocity, self.max_motor_velocity)
            right_speed = np.clip(float(action[1]), -self.max_motor_velocity, self.max_motor_velocity)
            self.left_motor.setVelocity(left_speed)
            self.right_motor.setVelocity(right_speed)
        except Exception as e:
            print(f"ERROR setting motor velocity: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, False, False, {}

        # Step simulation
        sim_result = self._webots_step()
        if sim_result == -1:
            obs = self._get_obs()
            return obs, 0, False, False, {}

        # Get observation
        obs = self._get_obs()

        # Check if robot is out of bounds
        out_of_bounds = False
        robot_position = None
        if self.robot_node:
            try:
                translation_field = self.robot_node.getField("translation")
                if translation_field:
                    pos = translation_field.getSFVec3f()
                    robot_position = pos
                    if abs(pos[0]) > 3.0 or abs(pos[2]) > 3.0:
                        out_of_bounds = True
            except:
                pass

        # Calculate reward
        min_lidar_normalized = np.min(obs) if obs.size > 0 else 1.0
        min_lidar_meters = min_lidar_normalized * self.lidar_max_range
        
        # TARGET REACHING REWARD
        target_reward = 0.0
        reached_target = False
        if robot_position and hasattr(self, 'target_position'):
            distance_to_target = np.sqrt(
                (robot_position[0] - self.target_position[0])**2 + 
                (robot_position[2] - self.target_position[2])**2
            )
            
            if distance_to_target < 0.3:
                target_reward = 100.0
                reached_target = True
                print(f"🎯 TARGET REACHED!")
            else:
                target_reward = (3.5 - distance_to_target) * 0.2
        
        # Collision penalty
        if min_lidar_meters < 0.05:
            crash_penalty = -10.0
        elif min_lidar_meters < 0.15:
            crash_penalty = -1.0
        else:
            crash_penalty = 0.0

        # Movement rewards
        avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
        forward_reward = (avg_speed / self.max_motor_velocity) * 0.5
        distance_bonus = min_lidar_normalized * 0.5
        differential = abs(left_speed - right_speed) / self.max_motor_velocity
        spin_penalty = -differential * 0.05
        
        # Total reward
        reward = forward_reward + crash_penalty + distance_bonus + spin_penalty + target_reward

        if out_of_bounds:
            reward -= 20.0

        # Termination conditions
        terminated = bool(min_lidar_meters < 0.05 or out_of_bounds or reached_target)
        truncated = bool(self.steps_taken >= self.max_steps)

        info = {'min_distance': float(min_lidar_meters), 'target_reward': float(target_reward)}
        return obs, reward, terminated, truncated, info


def main():
    print("=" * 60)
    print("--- TurtleBot3 RL Training ---")
    print("=" * 60)
    env = None
    
    print(f"Python version: {sys.version}")
    
    try:
        print("\n[1/4] Initializing environment...")
        env = TurtleBotGymEnvironment(max_episode_steps=1000)
        print("✓ Environment initialized successfully.")
        
        print("\n[2/4] Warming up simulation...")
        obs, info = env.reset()
        print(f"✓ Initial observation shape: {obs.shape}")
        
        print("\n[3/4] Testing manual steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, terminated={terminated}")
            if terminated or truncated:
                obs, info = env.reset()
        print("✓ Manual steps completed.")
        
        print("\n[4/4] Creating PPO model...")
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            tensorboard_log=None,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device='cpu'
        )
        print("✓ PPO model created.")
        
        print("\n" + "=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        
        saved_model_path = "ppo_turtlebot_model_interrupted"
        
        try:
            model.learn(total_timesteps=100_000)
            saved_model_path = "ppo_turtlebot_model"
            print("\n✓ Training completed!")
        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user.")
        except Exception as e:
            print(f"\n❌ ERROR during training: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n💾 Saving model to {saved_model_path}...")
        try:
            model.save(saved_model_path)
            print("✓ Model saved successfully.")
        except Exception as e_save:
            print(f"ERROR saving model: {e_save}")
        
        print("\n" + "="*60)
        print("Starting replay (10 episodes)")
        print("="*60)
        
        obs, info = env.reset()
        episode_count = 0
        max_replay_episodes = 10
        
        while episode_count < max_replay_episodes:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                episode_count += 1
                print(f"✓ Replay episode {episode_count}/{max_replay_episodes}")
                if episode_count < max_replay_episodes:
                    obs, info = env.reset()
        
        print("\n✅ All done!")
                
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("--- Script starting ---")
    main()