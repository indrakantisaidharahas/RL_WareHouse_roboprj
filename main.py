import os
import time

# Attempt import ManualEnv
try:
    from ManualEnv import ManualEnv
except ImportError as e:
    import sys
    print(f"FATAL: Failed to import ManualEnv: {e}")
    sys.exit(1)
except Exception as e:
    import sys
    print(f"FATAL: Unexpected error during ManualEnv import: {e}")
    sys.exit(1)

# Attempt import Stable Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    # from stable_baselines3.common.env_checker import check_env # Optional check
except ImportError:
    import sys
    print("FATAL: Stable-Baselines3 or Gymnasium not installed.")
    print('Run: "pip install stable-baselines3 gymnasium" in your venv_py310')
    sys.exit(1)

# --- Configuration ---
LOG_DIR = "./ppo_webots_logs/"
MODEL_SAVE_PATH = os.path.join(LOG_DIR, "ppo_epuck_final_model")
TOTAL_TIMESTEPS = 200_000 # Adjust as needed
SAVE_FREQ = 10_000
REPLAY_STEPS = 5_000
MAX_EPISODE_STEPS = 1000

os.makedirs(LOG_DIR, exist_ok=True)

# --- Environment Setup ---
print("INFO: Creating environment...")
try:
    # Set WEBOTS_HOME if needed
    if 'WEBOTS_HOME' not in os.environ:
         os.environ['WEBOTS_HOME'] = '/usr/local/webots'
         print(f"DEBUG: Setting WEBOTS_HOME to {os.environ['WEBOTS_HOME']}")

    env = ManualEnv(max_episode_steps=MAX_EPISODE_STEPS)
    print("INFO: Environment created successfully.")
except Exception as e:
    print(f"FATAL: Error creating environment: {e}")
    print("Ensure Webots is running, world loaded & playing, controller='<extern>', DEF name correct.")
    exit()

# --- Agent Training ---
checkpoint_callback = CheckpointCallback(save_freq=SAVE_FREQ, save_path=LOG_DIR, name_prefix="webots_ppo_model")

print("INFO: Creating PPO model...")
model = PPO(
    "MlpPolicy", env, verbose=1, tensorboard_log=LOG_DIR, device="auto",
    n_steps=1024, batch_size=64, n_epochs=10, gamma=0.99, gae_lambda=0.95,
    clip_range=0.2, ent_coef=0.0, learning_rate=3e-4,
)

print(f"INFO: Starting training for {TOTAL_TIMESTEPS} timesteps...")
print(f"INFO: Logs/models saved in: {LOG_DIR}")
start_time = time.time()
training_duration = 0
try:
    model.learn( total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback, log_interval=1, tb_log_name="PPO_Webots")
    training_duration = time.time() - start_time
    print(f"INFO: Training finished successfully in {training_duration:.2f}s ({training_duration/60:.2f}min).")
except KeyboardInterrupt:
    training_duration = time.time() - start_time
    print(f"\nWARN: Training interrupted after {training_duration:.2f}s.")
    MODEL_SAVE_PATH = os.path.join(LOG_DIR, "ppo_epuck_interrupted_model")
except Exception as e:
     training_duration = time.time() - start_time
     print(f"\nERROR: Training failed after {training_duration:.2f}s: {e}")
     import traceback
     traceback.print_exc()
     MODEL_SAVE_PATH = os.path.join(LOG_DIR, "ppo_epuck_error_model")

finally:
    # --- Save Model ---
    if 'model' in locals():
        print(f"INFO: Saving model to {MODEL_SAVE_PATH}...")
        try: model.save(MODEL_SAVE_PATH); print(f"INFO: Model saved.")
        except Exception as e: print(f"ERROR: Failed to save model: {e}")
    else: print("WARN: Model object not found, skipping save.")

    # --- Replay ---
    if 'model' in locals() and training_duration > 5: # Only replay if trained a bit
        print("\nINFO: Starting replay...")
        try:
            obs, info = env.reset()
            if obs is None: raise Exception("Reset failed before replay")
            episode_count = 0
            for i in range(REPLAY_STEPS):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    episode_count += 1
                    print(f"INFO: Replay episode {episode_count} finished. Resetting...")
                    obs, info = env.reset()
                    if obs is None: raise Exception("Reset failed during replay")
        except KeyboardInterrupt: print("WARN: Replay interrupted.")
        except Exception as e: print(f"ERROR during replay: {e}")
        finally: print("INFO: Replay finished.")
    else: print("INFO: Skipping replay.")

    # Optional: Close env (might close Webots)
    # env.close()
    print("INFO: Script finished.")