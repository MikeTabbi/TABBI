import pybullet as p
import pybullet_data
import time


def create_wall(position, half_extents, color=[0.7, 0.7, 0.7, 1]):
    """
    Create a wall at the given position.
    
    Args:
        position: [x, y, z] center of the wall
        half_extents: [half_length, half_width, half_height] of the box
        color: [r, g, b, a] color values from 0 to 1
    
    Returns:
        wall_id: the PyBullet body ID
    """
    # Create collision shape (what things bump into)
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents
    )
    
    # Create visual shape (what we see)
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )
    
    # Combine into a body and place in world
    wall_id = p.createMultiBody(
        baseMass=0,  # 0 mass = static (doesn't move)
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    
    return wall_id


def create_room(room_size=10, wall_height=2, wall_thickness=0.1):
    """
    Create a rectangular room with 4 walls.
    
    Args:
        room_size: length and width of room in meters
        wall_height: height of walls in meters
        wall_thickness: thickness of walls in meters
    
    Returns:
        list of wall IDs
    """
    half_size = room_size / 2
    half_height = wall_height / 2
    half_thickness = wall_thickness / 2
    
    walls = []
    
    # North wall (along X axis, at +Y edge)
    walls.append(create_wall(
        position=[0, half_size, half_height],
        half_extents=[half_size, half_thickness, half_height]
    ))
    
    # South wall (along X axis, at -Y edge)
    walls.append(create_wall(
        position=[0, -half_size, half_height],
        half_extents=[half_size, half_thickness, half_height]
    ))
    
    # East wall (along Y axis, at +X edge)
    walls.append(create_wall(
        position=[half_size, 0, half_height],
        half_extents=[half_thickness, half_size, half_height]
    ))
    
    # West wall (along Y axis, at -X edge)
    walls.append(create_wall(
        position=[-half_size, 0, half_height],
        half_extents=[half_thickness, half_size, half_height]
    ))
    
    return walls


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
    print(f"Created room with {len(walls)} walls")
    
    # Step 5: Load robot at center
    robot_start_pos = [0, 0, 0.5]
    robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
    robot_id = p.loadURDF("r2d2.urdf", robot_start_pos, robot_start_orientation)
    
    # Step 6: Set camera to bird's eye view
    p.resetDebugVisualizerCamera(
        cameraDistance=15,
        cameraYaw=0,
        cameraPitch=-89,  # Looking straight down
        cameraTargetPosition=[0, 0, 0]
    )
    
    print("Room environment created!")
    print("You should see a 10m x 10m room from above.")
    print("Use mouse to rotate view. Close window to exit.")
    
    # Step 7: Run simulation
    while True:
        p.stepSimulation()
        time.sleep(1/240)
        
        # Check if window was closed
        if not p.isConnected():
            break


if __name__ == "__main__":
    main()