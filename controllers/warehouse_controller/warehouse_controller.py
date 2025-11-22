"""Warehouse Controller - MATCHING YOUR 20x20 TRAINING"""
from controller import Supervisor
from stable_baselines3 import PPO
import numpy as np
import os

class Grid2DToWebots3D:
    def __init__(self):
        self.grid_size = 20  # MATCH YOUR TRAINING!
        self.cell_size = 0.75  # 15m / 20 cells = 0.75m per cell
        
    def grid_to_webots(self, gx, gy):
        # Map 20x20 grid to 15x15 meter Webots world
        wx = (gx - 10) * self.cell_size
        wz = (gy - 10) * self.cell_size
        return [wx, 0.2, wz]
    
    def webots_to_grid(self, pos):
        gx = int(round((pos[0] / self.cell_size) + 10))
        gy = int(round((pos[2] / self.cell_size) + 10))
        return max(0, min(19, gx)), max(0, min(19, gy))

class WebotsWarehouseController:
    def __init__(self, model_path="warehouse_final.zip"):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        print(f"\n📦 Loading: {os.path.basename(model_path)}")
        self.model = PPO.load(model_path)
        print("✅ Model loaded")
        
        self.converter = Grid2DToWebots3D()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.object_node = self.supervisor.getFromDef("OBJECT")
        
        # Training values
        self.grid_size = 20
        self.delivery_zone = [16, 16, 19, 19]  # From your training
        
        self.carrying = False
        self.delivered = False
        self.step_count = 0
        
    def get_observation(self):
        """EXACT 13-dim observation from your training"""
        rp = self.robot_node.getPosition()
        rgx, rgy = self.converter.webots_to_grid(rp)
        
        if not self.carrying:
            op = self.object_node.getPosition()
            ogx, ogy = self.converter.webots_to_grid(op)
        else:
            ogx, ogy = -1, -1
        
        # Delivery center
        dcx = (self.delivery_zone[0] + self.delivery_zone[2]) / 2.0
        dcy = (self.delivery_zone[1] + self.delivery_zone[3]) / 2.0
        
        # Distances (normalized by 20, not 15!)
        if not self.carrying:
            dist_obj = np.sqrt((rgx-ogx)**2 + (rgy-ogy)**2) / self.grid_size
            dist_del = 0.0
        else:
            dist_obj = 0.0
            dist_del = np.sqrt((rgx-dcx)**2 + (rgy-dcy)**2) / self.grid_size
        
        # Obstacle detection (4 directions)
        obs_up = 1 if rgy <= 0 else 0
        obs_down = 1 if rgy >= 19 else 0
        obs_left = 1 if rgx <= 0 else 0
        obs_right = 1 if rgx >= 19 else 0
        
        return np.array([
            rgx / self.grid_size,      # 0-1
            rgy / self.grid_size,      # 0-1
            ogx / self.grid_size if not self.carrying else -1,
            ogy / self.grid_size if not self.carrying else -1,
            dcx / self.grid_size,      # Delivery center x
            dcy / self.grid_size,      # Delivery center y
            dist_obj,
            dist_del,
            1.0 if self.carrying else 0.0,
            obs_up,
            obs_down,
            obs_left,
            obs_right
        ], dtype=np.float32)
    
    def teleport_action(self, action):
        """Teleport with boundary checking"""
        action = int(action)
        
        rp = self.robot_node.getPosition()
        cx, cy = self.converter.webots_to_grid(rp)
        
        # Action: 0=Up, 1=Down, 2=Left, 3=Right
        if action == 0:
            new_gx, new_gy = cx, cy - 1
        elif action == 1:
            new_gx, new_gy = cx, cy + 1
        elif action == 2:
            new_gx, new_gy = cx - 1, cy
        elif action == 3:
            new_gx, new_gy = cx + 1, cy
        else:
            return
        
        # Clamp to 20x20 grid
        new_gx = max(0, min(19, new_gx))
        new_gy = max(0, min(19, new_gy))
        
        # Teleport
        new_pos = self.converter.grid_to_webots(new_gx, new_gy)
        self.robot_node.getField('translation').setSFVec3f(new_pos)
        
        if self.step_count % 10 == 0:
            acts = ["UP", "DOWN", "LEFT", "RIGHT"]
            print(f"   {acts[action]}: ({cx},{cy}) → ({new_gx},{new_gy})")
    
    def check_pickup(self):
        """Check pickup (radius 1.5 cells from training)"""
        if self.carrying:
            return
        
        rp = self.robot_node.getPosition()
        op = self.object_node.getPosition()
        
        rgx, rgy = self.converter.webots_to_grid(rp)
        ogx, ogy = self.converter.webots_to_grid(op)
        
        dist = np.sqrt((rgx - ogx)**2 + (rgy - ogy)**2)
        
        if dist <= 1.5:
            self.carrying = True
            self.object_node.getField('translation').setSFVec3f([rp[0], rp[1]+0.5, rp[2]])
            print(f"\n   🎯 PICKED UP at Grid({rgx},{rgy})!\n")
    
    def check_delivery(self):
        """Check if in delivery zone [16-19, 16-19]"""
        if not self.carrying or self.delivered:
            return False
        
        rp = self.robot_node.getPosition()
        rgx, rgy = self.converter.webots_to_grid(rp)
        
        in_zone = (self.delivery_zone[0] <= rgx <= self.delivery_zone[2] and
                   self.delivery_zone[1] <= rgy <= self.delivery_zone[3])
        
        if in_zone:
            self.delivered = True
            print(f"\n   ⭐ DELIVERED at Grid({rgx},{rgy})! ⭐\n")
            return True
        
        return False
    
    def run(self):
        print("\n🤖 Warehouse Robot - 20×20 Grid (Matching Training)")
        print("=" * 60)
        
        # Set correct starting positions
        # Robot at Grid(2,2)
        start_pos = self.converter.grid_to_webots(2, 2)
        self.robot_node.getField('translation').setSFVec3f(start_pos)
        
        # Object at Grid(3,3) - or use spawn zones
        obj_pos = self.converter.grid_to_webots(3, 3)
        self.object_node.getField('translation').setSFVec3f([obj_pos[0], 0.3, obj_pos[2]])
        
        print(f"   Robot: Grid(2,2)")
        print(f"   Object: Grid(3,3)")
        print(f"   Delivery: Grid(16-19, 16-19)")
        print("=" * 60)
        
        while self.supervisor.step(self.timestep) != -1:
            self.step_count += 1
            
            if self.check_delivery():
                print(f"✅ SUCCESS in {self.step_count} steps!")
                break
            
            if self.carrying:
                rp = self.robot_node.getPosition()
                self.object_node.getField('translation').setSFVec3f([rp[0], rp[1]+0.5, rp[2]])
            
            self.check_pickup()
            
            obs = self.get_observation()
            action, _ = self.model.predict(obs, deterministic=True)
            self.teleport_action(action)
            
            # Visualization delay
            for _ in range(5):
                self.supervisor.step(self.timestep)
            
            if self.step_count % 20 == 0:
                rp = self.robot_node.getPosition()
                gx, gy = self.converter.webots_to_grid(rp)
                print(f"Step {self.step_count}: Grid({gx},{gy}), Carrying: {self.carrying}")
            
            if self.step_count > 500:
                print("⏱️ Timeout")
                break

if __name__ == "__main__":
    WebotsWarehouseController("warehouse_dstar_dqn").run()
