# minimal_test_controller.py
from controller import Robot

print("Minimal controller script started!") # Print message

robot = Robot()
timestep = int(robot.getBasicTimeStep())

left_motor = robot.getDevice("left wheel motor")
right_motor = robot.getDevice("right wheel motor")

if left_motor is None or right_motor is None:
    print("Error: Could not find motors!")
else:
    left_motor.setPosition(float('inf'))
    right_motor.setPosition(float('inf'))
    left_motor.setVelocity(1.0) # Move slowly
    right_motor.setVelocity(1.0)
    print("Motors found and set to move.")

# Main loop
count = 0
while robot.step(timestep) != -1:
    count += 1
    if count % 100 == 0: # Print message every 100 steps
         print(f"Minimal controller running... step {count}")
    if count > 500: # Run for a short time then stop
         break 

print("Minimal controller finished.")
left_motor.setVelocity(0.0)
right_motor.setVelocity(0.0)

# DO NOT CALL supervisor functions here
