import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import sys

# Add the Webots library path directly here
WEBOTS_LIB_PATH = '/usr/local/webots/lib/controller/python'
if WEBOTS_LIB_PATH not in sys.path:
    sys.path.append(WEBOTS_LIB_PATH)
print(f"DEBUG: Python Path includes: {WEBOTS_LIB_PATH}")

# Import from Webots library
try:
    from controller import Supervisor
except ImportError as e:
    print(f"FATAL ImportError: {e}")
    sys.exit(
        "ERROR: Could not import 'Supervisor'. Check WEBOTS_LIB_PATH and Python version (use 3.10 venv)."
    )

class ManualEnv(gym.Env):
    """ Webots E-puck Gym Env for external (<extern>) control """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self, max_episode_steps=1000, render_mode=None):
        super(ManualEnv, self).__init__()
        print("DEBUG: ManualEnv __init__ starting...")

        # Connect Supervisor
        try:
            self.supervisor = Supervisor()
        except Exception as e:
            print(f"FATAL: Could not initialize Supervisor: {e}")
            sys.exit(1)
        print("DEBUG: Supervisor initialized.")

        self.timestep = int(self.supervisor.getBasicTimeStep())
        self.max_steps = max_episode_steps
        self.steps_taken = 0
        self.render_mode = render_mode

        # Get robot node
        self.robot_node = self.supervisor.getFromDef("e-puck")
        if self.robot_node is None:
            print("FATAL: Robot node DEF 'e-puck' not found in world file.")
            self.supervisor.simulationQuit(1)
            sys.exit(1)
        print("DEBUG: Robot node 'e-puck' found.")

        # Get devices
        self.left_motor = self.robot_node.getDevice("left wheel motor")
        self.right_motor = self.robot_node.getDevice("right wheel motor")
        self.lidar = self.robot_node.getDevice("LDS-01")

        if self.left_motor is None or self.right_motor is None or self.lidar is None:
             print("FATAL: Could not get essential devices (motors/lidar). Check names.")
             self.supervisor.simulationQuit(1)
             sys.exit(1)
        print("DEBUG: Motors and Lidar found.")

        # Enable sensor
        self.lidar.enable(self.timestep)

        # Configure Motors
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.max_motor_velocity = self.left_motor.getMaxVelocity()

        # Define Action Space
        self.action_space = spaces.Box(
            low=-self.max_motor_velocity, high=self.max_motor_velocity, shape=(2,), dtype=np.float32
        )

        # Define Observation Space
        try:
            self.lidar_points = self.lidar.getHorizontalResolution()
            self.lidar_max_range = self.lidar.getMaxRange()
            if self.lidar_points <= 0: raise ValueError("Lidar points invalid")
        except Exception as e:
            print(f"FATAL: Error getting lidar properties: {e}. Using defaults.")
            self.lidar_points = 360; self.lidar_max_range = 3.5 # Fallback defaults

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.lidar_points,), dtype=np.float32
        )

        print("DEBUG: Environment Initialized Successfully.")
        print(f"DEBUG: Action Space: {self.action_space}")
        print(f"DEBUG: Observation Space: {self.observation_space} ({self.lidar_points} points)")

    def _get_obs(self):
        lidar_values = self.lidar.getRangeImage()
        max_wait = 5
        count = 0
        while lidar_values is None and count < max_wait:
            if self.supervisor.step(self.timestep) == -1: return None
            lidar_values = self.lidar.getRangeImage()
            count += 1

        if lidar_values is None:
            print("WARN: _get_obs: Lidar data is None after waiting.")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        lidar_values = np.array(lidar_values) / self.lidar_max_range
        lidar_values = np.clip(lidar_values, 0.0, 1.0)
        lidar_values[~np.isfinite(lidar_values)] = 1.0

        if len(lidar_values) != self.lidar_points:
            print(f"WARN: _get_obs: Lidar shape mismatch {len(lidar_values)} vs {self.lidar_points}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        return lidar_values.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("DEBUG: reset() called.")
        self.supervisor.simulationResetPhysics()
        self.supervisor.simulationReset()
        self.steps_taken = 0
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        # Randomize position
        translation_field = self.robot_node.getField("translation")
        rotation_field = self.robot_node.getField("rotation")
        if translation_field and rotation_field:
            new_x = np.random.uniform(-1.5, 1.5); new_z = np.random.uniform(-1.5, 1.5)
            new_rot = np.random.uniform(-np.pi, np.pi)
            translation_field.setSFVec3f([new_x, 0.02, new_z])
            rotation_field.setSFRotation([0, 1, 0, new_rot])
            self.robot_node.resetPhysics()
            print(f"DEBUG: Robot randomized to x={new_x:.2f}, z={new_z:.2f}")

        # Initial step and observation
        if self.supervisor.step(self.timestep) == -1:
            print("WARN: Sim quit during reset step.")
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        obs = self._get_obs()
        # Stabilize
        for _ in range(3):
            if obs is None or np.all(obs==0):
                 if self.supervisor.step(self.timestep) == -1: break
                 obs = self._get_obs()
            else: break
        if obs is None: obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        print("DEBUG: reset() finished.")
        return obs, {}

    def step(self, action):
        self.steps_taken += 1
        left_speed = np.clip(float(action[0]), -self.max_motor_velocity, self.max_motor_velocity)
        right_speed = np.clip(float(action[1]), -self.max_motor_velocity, self.max_motor_velocity)
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

        if self.supervisor.step(self.timestep) == -1:
            print("WARN: Sim quit during step.")
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, True, {}

        obs = self._get_obs()
        if obs is None:
            print("WARN: Sim quit after step before getting obs.")
            return np.zeros(self.observation_space.shape, dtype=np.float32), 0, True, True, {}

        # Reward calculation
        min_lidar = np.min(obs) if obs.size > 0 else 1.0
        crash_penalty = 0.0
        if min_lidar < 0.05: crash_penalty = -10.0
        elif min_lidar < 0.15: crash_penalty = -2.0 * (1.0 - (min_lidar / 0.15))
        average_speed = (abs(left_speed) + abs(right_speed)) / 2.0
        speed_reward = average_speed / self.max_motor_velocity * 0.05
        reward = speed_reward + crash_penalty

        # Done flags
        terminated = bool(crash_penalty <= -10.0)
        truncated = bool(self.steps_taken >= self.max_steps)

        # print(f"DEBUG Step: {self.steps_taken}, Reward: {reward:.3f}, Term: {terminated}, Trunc: {truncated}") # Verbose
        return obs, reward, terminated, truncated, {}