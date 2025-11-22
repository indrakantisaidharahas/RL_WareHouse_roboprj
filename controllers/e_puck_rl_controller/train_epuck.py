from stable_baselines3 import PPO
from e_puck_rl_controller import EPuckGymEnvironment

env = EPuckGymEnvironment()
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_epuck")
