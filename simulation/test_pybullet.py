import pybullet as p
import pybullet_data
import time


def main():
    # Step 1: Connect to PyBullet in GUI mode (opens a window)
    physics_client = p.connect(p.GUI)
    
    # Step 2: Tell PyBullet where to find built-in objects
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    # Step 3: Set gravity (things fall down)
    p.setGravity(0, 0, -9.81)
    
    # Step 4: Load a floor
    plane_id = p.loadURDF("plane.urdf")
    
    # Step 5: Load a robot
    robot_start_pos = [0, 0, 0.5]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("r2d2.urdf", robot_start_pos, robot_start_orientation)
    
    # Step 6: Run simulation
    print("PyBullet simulation running!")
    print("You should see a window with R2D2 on a gray plane.")
    print("It will run for 10 seconds, then close.")
    
    for i in range(240 * 10):
        p.stepSimulation()
        time.sleep(1/240)
    
    # Step 7: Close
    p.disconnect()
    print("Simulation complete!")


if __name__ == "__main__":
    main()