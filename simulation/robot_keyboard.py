import pybullet as p
import pybullet_data
import time

# Import our room creation function
from room_environment import create_room


def main():
    # Step 1: Connect to PyBullet
    physics_client = p.connect(p.GUI)
    
    # Step 2: Set up environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Step 3: Load floor
    plane_id = p.loadURDF("plane.urdf")
    
    # Step 4: Create room
    walls = create_room(room_size=10, wall_height=2, wall_thickness=0.1)
    
    # Step 5: Load Husky robot (4-wheeled robot)
    robot_start_pos = [0, 0, 0.2]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("husky/husky.urdf", robot_start_pos, robot_start_orientation)
    
    # Step 6: Find the wheel joints
    # Husky has 4 wheels - we need their joint indices
    num_joints = p.getNumJoints(robot_id)
    wheel_joints = []
    
    print("Robot joints:")
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        print(f"  Joint {i}: {joint_name}")
        
        # Find wheel joints (they have 'wheel' in the name)
        if 'wheel' in joint_name.lower():
            wheel_joints.append(i)
    
    print(f"Wheel joints: {wheel_joints}")
    
    # Husky wheels: [front_left, front_right, rear_left, rear_right]
    front_left = 2
    front_right = 3
    rear_left = 4
    rear_right = 5
    
    # Step 7: Set camera
    p.resetDebugVisualizerCamera(
        cameraDistance=12,
        cameraYaw=0,
        cameraPitch=-60,
        cameraTargetPosition=[0, 0, 0]
    )
    
    # Movement speeds
    max_velocity = 10
    turn_velocity = 5
    
    print("\n=== CONTROLS ===")
    print("UP ARROW    : Move forward")
    print("DOWN ARROW  : Move backward")
    print("LEFT ARROW  : Turn left")
    print("RIGHT ARROW : Turn right")
    print("Close window to exit")
    print("================\n")
    
    # Step 8: Main loop
    while p.isConnected():
        # Get keyboard events
        keys = p.getKeyboardEvents()
        
        # Initialize wheel velocities
        left_velocity = 0
        right_velocity = 0
        
        # UP arrow - move forward
        if p.B3G_UP_ARROW in keys:
            left_velocity = max_velocity
            right_velocity = max_velocity
        
        # DOWN arrow - move backward
        if p.B3G_DOWN_ARROW in keys:
            left_velocity = -max_velocity
            right_velocity = -max_velocity
        
        # LEFT arrow - turn left
        if p.B3G_LEFT_ARROW in keys:
            left_velocity = -turn_velocity
            right_velocity = turn_velocity
        
        # RIGHT arrow - turn right
        if p.B3G_RIGHT_ARROW in keys:
            left_velocity = turn_velocity
            right_velocity = -turn_velocity
        
        # Apply velocities to wheels
        p.setJointMotorControl2(robot_id, front_left, p.VELOCITY_CONTROL, targetVelocity=left_velocity)
        p.setJointMotorControl2(robot_id, rear_left, p.VELOCITY_CONTROL, targetVelocity=left_velocity)
        p.setJointMotorControl2(robot_id, front_right, p.VELOCITY_CONTROL, targetVelocity=right_velocity)
        p.setJointMotorControl2(robot_id, rear_right, p.VELOCITY_CONTROL, targetVelocity=right_velocity)
        
        # Step simulation
        p.stepSimulation()
        time.sleep(1/240)
    
    print("Simulation ended.")


if __name__ == "__main__":
    main()