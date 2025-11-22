"""
Create Comparison GIF showing Learning Progression
Shows: Untrained → Early Training → Fully Trained

Run after training: python create_comparison.py
"""

import gymnasium as gym
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import DQN
import sys

# Import the environment from your main file
try:
    from robot_pathplanning_v2 import RobotPathPlanningEnv
except:
    print("⚠️  Make sure robot_pathplanning_v2.py is in the same directory!")
    sys.exit(1)


def create_untrained_episode(env, max_steps=50):
    """Simulate untrained robot (random actions)"""
    frames = []
    obs, info = env.reset()
    
    for step in range(max_steps):
        # Random action (untrained behavior)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        if terminated or truncated:
            break
    
    return frames


def create_trained_episode(model, env, max_steps=50):
    """Run trained model"""
    frames = []
    obs, info = env.reset()
    
    for step in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        
        if terminated or truncated:
            break
    
    return frames


def add_label(frame, text, color=(255, 255, 255), bg_color=(0, 0, 0)):
    """Add text label to frame"""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Try to use a nice font, fallback to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 24)
    except:
        font = ImageFont.load_default()
    
    # Get text size
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Draw background rectangle
    padding = 10
    rect_coords = [
        (img.width - text_width - padding * 2, 0),
        (img.width, text_height + padding * 2)
    ]
    draw.rectangle(rect_coords, fill=bg_color)
    
    # Draw text
    text_position = (img.width - text_width - padding, padding)
    draw.text(text_position, text, fill=color, font=font)
    
    return np.array(img)


def create_side_by_side_frame(frame1, frame2, label1, label2):
    """Create side-by-side comparison frame"""
    # Add labels
    frame1 = add_label(frame1, label1, color=(255, 0, 0), bg_color=(50, 0, 0))
    frame2 = add_label(frame2, label2, color=(0, 255, 0), bg_color=(0, 50, 0))
    
    # Combine side by side
    h, w = frame1.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = frame1
    combined[:, w:] = frame2
    
    return combined


def create_comparison_gif():
    """Main function to create comparison GIF"""
    print("🎬 Creating Learning Progress Comparison GIF...")
    print("=" * 60)
    
    # Load trained model
    try:
        model = DQN.load("robot_pathplanning_dqn.zip")
        print("✅ Loaded trained model")
    except:
        print("❌ Could not load robot_pathplanning_dqn.zip")
        print("   Make sure you've trained the model first!")
        return
    
    # Create environment
    env = RobotPathPlanningEnv(grid_size=15, num_obstacles=15, render_mode='rgb_array')
    
    # Generate episodes
    print("\n📹 Generating untrained episode (random actions)...")
    untrained_frames = create_untrained_episode(env, max_steps=50)
    
    print("📹 Generating trained episode (learned policy)...")
    trained_frames = create_trained_episode(model, env, max_steps=50)
    
    # Create comparison frames
    print("\n🎨 Creating side-by-side comparison...")
    comparison_frames = []
    
    max_length = max(len(untrained_frames), len(trained_frames))
    
    for i in range(max_length):
        # Get frames (repeat last frame if one sequence is shorter)
        frame1 = untrained_frames[min(i, len(untrained_frames) - 1)]
        frame2 = trained_frames[min(i, len(trained_frames) - 1)]
        
        # Create side-by-side
        combined = create_side_by_side_frame(
            frame1, frame2,
            "UNTRAINED (Random)", "TRAINED (DQN)"
        )
        comparison_frames.append(combined)
    
    # Save as GIF
    print("\n💾 Saving GIF...")
    images = [Image.fromarray(frame) for frame in comparison_frames]
    
    images[0].save(
        'learning_comparison.gif',
        save_all=True,
        append_images=images[1:],
        duration=150,  # 150ms per frame
        loop=0
    )
    
    print("✅ Comparison GIF saved as: learning_comparison.gif")
    print(f"   - Untrained episode: {len(untrained_frames)} frames")
    print(f"   - Trained episode: {len(trained_frames)} frames")
    print(f"   - Total comparison: {len(comparison_frames)} frames")
    
    env.close()
    
    # Also create individual GIFs
    print("\n📦 Creating individual GIFs...")
    
    # Untrained GIF
    labeled_untrained = [add_label(f, "UNTRAINED", (255, 0, 0), (50, 0, 0)) for f in untrained_frames]
    images_untrained = [Image.fromarray(f) for f in labeled_untrained]
    images_untrained[0].save(
        'untrained_robot.gif',
        save_all=True,
        append_images=images_untrained[1:],
        duration=150,
        loop=0
    )
    print("✅ Untrained GIF saved as: untrained_robot.gif")
    
    # Trained GIF (already exists as robot_demo.gif, but create labeled version)
    labeled_trained = [add_label(f, "TRAINED", (0, 255, 0), (0, 50, 0)) for f in trained_frames]
    images_trained = [Image.fromarray(f) for f in labeled_trained]
    images_trained[0].save(
        'trained_robot.gif',
        save_all=True,
        append_images=images_trained[1:],
        duration=150,
        loop=0
    )
    print("✅ Trained GIF saved as: trained_robot.gif")
    
    print("\n" + "=" * 60)
    print("🎉 Done! Created 3 GIFs:")
    print("   1. learning_comparison.gif (side-by-side)")
    print("   2. untrained_robot.gif (before learning)")
    print("   3. trained_robot.gif (after learning)")
    print("\nView them with: xdg-open learning_comparison.gif")


if __name__ == "__main__":
    create_comparison_gif()
