# e_puck_rl_controller.py (Fixed Version - No Finally Block)
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
    from stable_baselines3.common.env_checker import check_env
except ImportError:
    print('FATAL ERROR: Stable-Baselines3 or Gymnasium not installed.')
    sys.exit(1)


class EPuckGymEnvironment(Supervisor, gym.Env):
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, max_episode_steps=1000, render_mode=None):
        print("DEBUG: Entering EPuckGymEnvironment.__init__")
        try:
            Supervisor.__init__(self)
        except Exception as e:
            print(f"FATAL ERROR during Supervisor.__init__(): {e}")
            sys.exit(1)
        print("DEBUG: Supervisor.__init__() called.")

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
        
        # Device parameters (defaults)
        self.max_motor_velocity = 6.28
        self.lidar_points = 360
        self.lidar_max_range = 3.5

        print(f"DEBUG: Timestep set to {self.timestep}")

        # --- Define Spaces (Use defaults initially) ---
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
            id='WebotsEPuckEnv-v0', 
            max_episode_steps=max_episode_steps
        )
        print("DEBUG: __init__ finished successfully.")

    def _webots_step(self):
        """Wrapper for Webots simulation step to avoid conflicts."""
        return Supervisor.step(self, self.timestep)

    def _setup_devices(self):
        """Get handles to devices - called only once in reset."""
        print("DEBUG: _setup_devices() called.")
        
        # Get the robot node (try both methods)
        self.robot_node = self.getFromDef("e-puck")
        if self.robot_node is None:
            print("WARNING: Robot DEF 'e-puck' not found. Using self as robot.")
            self.robot_node = self.getSelf()
        else:
            print(f"DEBUG: Found robot node with DEF 'e-puck'")
        
        if self.robot_node is None:
            print("ERROR: Could not get robot node!")
            return False

        # Get devices directly (since controller IS the robot)
        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.lidar = self.getDevice("LDS-01")

        if self.left_motor is None or self.right_motor is None:
            print("ERROR: Could not find motor devices!")
            print("Available devices:", [self.getDeviceName(i) for i in range(self.getNumberOfDevices())])
            return False
            
        if self.lidar is None:
            print("ERROR: Could not find LDS-01 lidar!")
            print("Available devices:", [self.getDeviceName(i) for i in range(self.getNumberOfDevices())])
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

        # Update space definitions with actual values
        self.max_motor_velocity = self.left_motor.getMaxVelocity()
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
        
        # CRITICAL FIX: The 0.008 readings are because of the lidar detecting the robot itself
        # Replace all readings below a reasonable threshold with max range
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
        # Reduced debug output
        super().reset(seed=seed)

        # Setup devices on first reset
        if not self.devices_initialized:
            print("DEBUG: First reset - setting up devices...")
            if not self._setup_devices():
                print("ERROR: Device setup failed!")
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}

        # DON'T reset physics during training - just reposition robot
        # Only do full reset on first call
        if self.steps_taken == 0 and not hasattr(self, '_first_reset_done'):
            self.simulationResetPhysics()
            if self._webots_step() == -1:
                print("ERROR: Sim quit during reset")
                return np.zeros(self.observation_space.shape, dtype=np.float32), {}
            self._first_reset_done = True

        # Reset internals
        self.steps_taken = 0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Reposition robot instead of physics reset
        if self.robot_node:
            translation_field = self.robot_node.getField("translation")
            rotation_field = self.robot_node.getField("rotation")
            
            if translation_field and rotation_field:
                try:
                    # Arena is 5x5 meters, keep robot safely inside (leave 0.5m margin from walls)
                    arena_size = 2.0  # Safe area = 4x4 meters (keeping 0.5m from each wall)
                    new_x = np.random.uniform(-arena_size, arena_size)
                    new_z = np.random.uniform(-arena_size, arena_size)
                    new_rot = np.random.uniform(-np.pi, np.pi)
                    
                    # IMPORTANT: Keep Y at 0.0 (ground level), not 0.02
                    # The robot's origin is at the bottom, so Y=0 is correct
                    translation_field.setSFVec3f([new_x, 0.0, new_z])
                    rotation_field.setSFRotation([0, 1, 0, new_rot])
                    
                    # Reset velocity
                    velocity_field = self.robot_node.getField("velocity")
                    if velocity_field:
                        velocity_field.setSFVec3f([0, 0, 0])
                    
                    # Reset angular velocity
                    angular_velocity_field = self.robot_node.getField("angularVelocity")
                    if angular_velocity_field:
                        angular_velocity_field.setSFVec3f([0, 0, 0])
                    
                except Exception as e:
                    print(f"WARNING: Could not reposition: {e}")

        # Step a few times for stabilization
        for _ in range(3):
            if self._webots_step() == -1:
                break

        # Get initial observation
        obs = self._get_obs()
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
            print("WARNING: Simulation ended during step")
            obs = self._get_obs()
            return obs, 0, False, False, {}

        # Get observation
        obs = self._get_obs()

        # Check if robot is out of bounds
        out_of_bounds = False
        if self.robot_node:
            try:
                translation_field = self.robot_node.getField("translation")
                if translation_field:
                    pos = translation_field.getSFVec3f()
                    # Arena is 5x5, walls are at ±2.5, allow robot to get close but not through
                    # E-puck radius is ~0.037m, so check at ±2.45
                    if abs(pos[0]) > 2.45 or abs(pos[2]) > 2.45:
                        out_of_bounds = True
                        print(f"WARNING: Robot out of bounds at ({pos[0]:.2f}, {pos[2]:.2f})")
            except:
                pass

        # Calculate reward
        min_lidar = np.min(obs) if obs.size > 0 else 1.0
        
        # Penalty for being too close to obstacles (more lenient threshold)
        if min_lidar < 0.02:
            crash_penalty = -10.0
        elif min_lidar < 0.1:
            crash_penalty = -1.0
        else:
            crash_penalty = 0.0
        
        # Reward for moving forward
        avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
        forward_reward = (avg_speed / self.max_motor_velocity) * 0.2
        
        # Bonus for maintaining distance from obstacles
        distance_bonus = min_lidar * 0.3
        
        # Penalty for spinning in place
        differential = abs(left_speed - right_speed) / self.max_motor_velocity
        spin_penalty = -differential * 0.1
        
        reward = forward_reward + crash_penalty + distance_bonus + spin_penalty

        # Penalty for going out of bounds
        if out_of_bounds:
            reward -= 20.0

        # Check termination conditions
        terminated = bool(min_lidar < 0.02 or out_of_bounds)
        truncated = bool(self.steps_taken >= self.max_steps)

        info = {'min_distance': float(min_lidar)}
        return obs, reward, terminated, truncated, info


def main():
    print("=" * 60)
    print("--- Starting main() ---")
    print("=" * 60)
    env = None
    
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    try:
        print("\n[1/4] Initializing environment...")
        env = EPuckGymEnvironment(max_episode_steps=1000)
        print("✓ Environment initialized successfully.")
        
        print("\n[2/4] Warming up simulation...")
        obs, info = env.reset()
        print(f"✓ Initial observation shape: {obs.shape}")
        print(f"✓ Observation sample (first 10 values): {obs[:10]}")
        
        print("\n[3/4] Testing manual steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.3f}, terminated={terminated}, truncated={truncated}")
            if terminated or truncated:
                obs, info = env.reset()
        print("✓ Manual steps completed successfully.")
        
        print("\n[4/4] Creating PPO model...")
        model = PPO(
            'MlpPolicy', 
            env, 
            verbose=1, 
            tensorboard_log=None,  # Disable tensorboard to avoid dependency issues
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
        print("This will take a while. Watch the robot in Webots!")
        print("Training stats will appear below:")
        print("=" * 60)
        
        saved_model_path = "ppo_epuck_model_interrupted"
        
        try:
            model.learn(total_timesteps=200_000)
            saved_model_path = "ppo_epuck_model"
            print("\n" + "="*60)
            print("Training completed successfully!")
            print("="*60)
        except KeyboardInterrupt:
            print("\n" + "="*60)
            print("Training interrupted by user.")
            print("="*60)
        except Exception as e:
            print("\n" + "="*60)
            print(f"ERROR during training: {e}")
            print("="*60)
            import traceback
            traceback.print_exc()
        
        # Save model after training
        print(f"\nSaving model to {saved_model_path}...")
        try:
            model.save(saved_model_path)
            print("✓ Model saved successfully.")
        except Exception as e_save:
            print(f"ERROR saving model: {e_save}")
        
        print("\n" + "="*60)
        print("Starting replay (watch the trained robot!)")
        print("Will replay 10 episodes then stop")
        print("="*60)
        
        obs, info = env.reset()
        episode_count = 0
        max_replay_episodes = 10
        
        while episode_count < max_replay_episodes:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                episode_count += 1
                print(f"✓ Replay episode {episode_count}/{max_replay_episodes} finished.")
                if episode_count < max_replay_episodes:
                    obs, info = env.reset()
        
        print("\n" + "="*60)
        print("Replay complete! Training and testing finished.")
        print("="*60)
                
    except KeyboardInterrupt:
        print("\nExecution interrupted by user.")
    except Exception as e:
        print(f"\nERROR during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    print("--- Script starting ---")
    os.environ['WEBOTS_HOME'] = '/usr/local/webots'
    main()