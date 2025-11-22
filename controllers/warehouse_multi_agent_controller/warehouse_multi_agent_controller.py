"""
Multi-Delivery Warehouse Controller - FINAL VERSION
5 sequential deliveries with robot reset
"""

from controller import Supervisor
from stable_baselines3 import PPO
import numpy as np
import os


class Grid2DToWebots3D:
    def __init__(self):
        self.grid_size = 20
        self.cell_size = 0.75
        
    def grid_to_webots(self, gx, gy):
        wx = (gx - 10) * self.cell_size
        wz = (gy - 10) * self.cell_size
        return [wx, 0.2, wz]
    
    def webots_to_grid(self, pos):
        gx = int(round((pos[0] / self.cell_size) + 10))
        gy = int(round((pos[2] / self.cell_size) + 10))
        return max(0, min(19, gx)), max(0, min(19, gy))


class MultiDeliveryController:
    def __init__(self, model_path="warehouse_dstar_dqn"):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())
        
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), model_path)
        
        print(f"📦 Loading: {os.path.basename(model_path)}")
        self.model = PPO.load(model_path)
        print("✅ PPO loaded\n")
        
        self.converter = Grid2DToWebots3D()
        self.robot_node = self.supervisor.getFromDef("ROBOT")
        self.delivery_zone = [16, 16, 19, 19]
        
        self.current_object = None
        self.carrying = False
        self.step_count = 0
        self.total_steps = 0
        self.delivered_count = 0
        self.total_deliveries = 5
        
    def spawn_new_object(self):
        """Spawn at Grid(3,3) - same as training"""
        gx, gy = 3, 3
        pos = self.converter.grid_to_webots(gx, gy)
        
        object_def = f"""
        DEF OBJECT{self.delivered_count + 1} Solid {{
          translation {pos[0]} 0.3 {pos[2]}
          children [
            Shape {{
              appearance PBRAppearance {{
                baseColor 1 0.9 0
                metalness 0.4
              }}
              geometry Box {{ size 0.5 0.5 0.5 }}
            }}
          ]
          name "box{self.delivered_count + 1}"
          boundingObject Box {{ size 0.5 0.5 0.5 }}
        }}
        """
        
        root = self.supervisor.getRoot()
        root.getField("children").importMFNodeFromString(-1, object_def)
        new_obj = self.supervisor.getFromDef(f"OBJECT{self.delivered_count + 1}")
        
        if new_obj:
            print(f"📦 Package #{self.delivered_count + 1} spawned at Grid({gx},{gy})")
            return new_obj
        return None
    
    def get_observation(self):
        rp = self.robot_node.getPosition()
        rgx, rgy = self.converter.webots_to_grid(rp)
        
        if not self.carrying and self.current_object:
            op = self.current_object.getPosition()
            ogx, ogy = self.converter.webots_to_grid(op)
        else:
            ogx, ogy = -1, -1
        
        dcx = (self.delivery_zone[0] + self.delivery_zone[2]) / 2.0
        dcy = (self.delivery_zone[1] + self.delivery_zone[3]) / 2.0
        
        if not self.carrying:
            dist_obj = np.sqrt((rgx-ogx)**2 + (rgy-ogy)**2) / 20 if self.current_object else 0
            dist_del = 0.0
        else:
            dist_obj = 0.0
            dist_del = np.sqrt((rgx-dcx)**2 + (rgy-dcy)**2) / 20
        
        return np.array([
            rgx / 20, rgy / 20,
            ogx / 20 if not self.carrying else -1,
            ogy / 20 if not self.carrying else -1,
            dcx / 20, dcy / 20,
            dist_obj, dist_del,
            1.0 if self.carrying else 0.0,
            1 if rgy <= 0 else 0, 1 if rgy >= 19 else 0,
            1 if rgx <= 0 else 0, 1 if rgx >= 19 else 0
        ], dtype=np.float32)
    
    def execute_action(self, action):
        rp = self.robot_node.getPosition()
        cx, cy = self.converter.webots_to_grid(rp)
        
        moves = {0: (0, -1), 1: (0, 1), 2: (-1, 0), 3: (1, 0)}
        dx, dy = moves.get(int(action), (0, 0))
        new_gx = max(0, min(19, cx + dx))
        new_gy = max(0, min(19, cy + dy))
        
        target_pos = self.converter.grid_to_webots(new_gx, new_gy)
        self.robot_node.getField('translation').setSFVec3f(target_pos)
        
        if self.carrying and self.current_object:
            self.current_object.getField('translation').setSFVec3f([target_pos[0], 0.7, target_pos[2]])
    
    def check_pickup(self):
        if self.carrying or not self.current_object:
            return
        
        rp = self.robot_node.getPosition()
        op = self.current_object.getPosition()
        rgx, rgy = self.converter.webots_to_grid(rp)
        ogx, ogy = self.converter.webots_to_grid(op)
        
        if abs(rgx - ogx) <= 1 and abs(rgy - ogy) <= 1:
            self.carrying = True
            print(f"🎯 PICKED UP Package #{self.delivered_count + 1}!")
    
    def check_delivery(self):
        if not self.carrying:
            return False
        
        rp = self.robot_node.getPosition()
        rgx, rgy = self.converter.webots_to_grid(rp)
        
        if (self.delivery_zone[0] <= rgx <= self.delivery_zone[2] and
            self.delivery_zone[1] <= rgy <= self.delivery_zone[3]):
            
            self.carrying = False
            self.delivered_count += 1
            
            if self.current_object:
                self.current_object.remove()
                self.current_object = None
            
            print(f"⭐ DELIVERED Package #{self.delivered_count}! ({self.step_count} steps)")
            
            # Reset robot to starting position
            start_pos = self.converter.grid_to_webots(2, 2)
            self.robot_node.getField('translation').setSFVec3f(start_pos)
            
            # Reset step counter for next delivery
            self.step_count = 0
            print(f"   🔄 Robot reset to Grid(2,2)")
            
            return True
        return False
    
    def run(self):
        print("🤖 MULTI-DELIVERY SYSTEM - 5 Packages")
        print("=" * 60)
        
        # First delivery with existing object
        self.current_object = self.supervisor.getFromDef("OBJECT")
        
        while self.supervisor.step(self.timestep) != -1:
            self.step_count += 1
            self.total_steps += 1
            
            self.check_pickup()
            
            if self.check_delivery():
                # Spawn next object
                if self.delivered_count < self.total_deliveries:
                    self.current_object = self.spawn_new_object()
                else:
                    print(f"\n✅ ALL {self.total_deliveries} DELIVERIES COMPLETE!")
                    print(f"   Total steps: {self.total_steps}")
                    print("=" * 60)
                    break
            
            if self.current_object or self.carrying:
                obs = self.get_observation()
                action, _ = self.model.predict(obs, deterministic=True)
                self.execute_action(action)
            
            # Timeout per delivery
            if self.step_count > 100:
                print(f"\n⏱️ Delivery timeout - {self.delivered_count}/{self.total_deliveries} delivered")
                break


if __name__ == "__main__":
    MultiDeliveryController("warehouse_dstar_dqn").run()
