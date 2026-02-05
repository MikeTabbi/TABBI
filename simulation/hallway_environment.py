import pybullet as p
import pybullet_data
import time
import math
import random


def create_wall(position, half_extents, color=[0.8, 0.8, 0.8, 1]):
    """
    Create a wall at the given position.
    
    Args:
        position: [x, y, z] center of the wall
        half_extents: [half_length, half_width, half_height] of the box
        color: [r, g, b, a] color values from 0 to 1
    
    Returns:
        wall_id: the PyBullet body ID
    """
    collision_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents
    )
    
    visual_shape = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=half_extents,
        rgbaColor=color
    )
    
    wall_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=position
    )
    
    return wall_id


def create_island(center_x, center_y, width, depth, wall_height=2.0, wall_thickness=0.1):
    """
    Create a solid rectangular island (robot cannot pass through).
    
    Args:
        center_x, center_y: Center position of island
        width: Size in X direction
        depth: Size in Y direction
        wall_height: Height of walls
        wall_thickness: Thickness of walls
    
    Returns:
        List of wall IDs
    """
    walls = []
    half_width = width / 2
    half_depth = depth / 2
    half_height = wall_height / 2
    half_thick = wall_thickness / 2
    
    # North wall
    walls.append(create_wall(
        position=[center_x, center_y + half_depth, half_height],
        half_extents=[half_width, half_thick, half_height]
    ))
    
    # South wall
    walls.append(create_wall(
        position=[center_x, center_y - half_depth, half_height],
        half_extents=[half_width, half_thick, half_height]
    ))
    
    # East wall
    walls.append(create_wall(
        position=[center_x + half_width, center_y, half_height],
        half_extents=[half_thick, half_depth, half_height]
    ))
    
    # West wall
    walls.append(create_wall(
        position=[center_x - half_width, center_y, half_height],
        half_extents=[half_thick, half_depth, half_height]
    ))
    
    # Fill the island (so robot can't go through)
    fill_shape = p.createCollisionShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half_width - half_thick, half_depth - half_thick, half_height]
    )
    fill_visual = p.createVisualShape(
        shapeType=p.GEOM_BOX,
        halfExtents=[half_width - half_thick, half_depth - half_thick, half_height],
        rgbaColor=[0.6, 0.6, 0.6, 1]
    )
    walls.append(p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=fill_shape,
        baseVisualShapeIndex=fill_visual,
        basePosition=[center_x, center_y, half_height]
    ))
    
    return walls


def create_hallway_environment(env_size=20, hallway_width=3, wall_height=2.0):
    """
    Create a hallway environment with intersections.
    
    Args:
        env_size: Total size of environment (square)
        hallway_width: Width of hallways
        wall_height: Height of walls
    
    Returns:
        Dictionary with wall IDs and valid positions
    """
    walls = []
    half_size = env_size / 2
    half_height = wall_height / 2
    wall_thickness = 0.1
    half_thick = wall_thickness / 2
    
    # Create outer boundary
    # North wall
    walls.append(create_wall(
        position=[0, half_size, half_height],
        half_extents=[half_size, half_thick, half_height]
    ))
    # South wall
    walls.append(create_wall(
        position=[0, -half_size, half_height],
        half_extents=[half_size, half_thick, half_height]
    ))
    # East wall
    walls.append(create_wall(
        position=[half_size, 0, half_height],
        half_extents=[half_thick, half_size, half_height]
    ))
    # West wall
    walls.append(create_wall(
        position=[-half_size, 0, half_height],
        half_extents=[half_thick, half_size, half_height]
    ))
    
    # Calculate island size based on hallway width
    # Islands fill the space between hallways
    island_size = (env_size - 3 * hallway_width) / 2
    
    # Island offset from center
    island_offset = (island_size + hallway_width) / 2
    
    # Create 4 islands (creates + shaped hallway pattern)
    # Top-left island
    island_walls = create_island(-island_offset, island_offset, island_size, island_size, wall_height)
    walls.extend(island_walls)
    
    # Top-right island
    island_walls = create_island(island_offset, island_offset, island_size, island_size, wall_height)
    walls.extend(island_walls)
    
    # Bottom-left island
    island_walls = create_island(-island_offset, -island_offset, island_size, island_size, wall_height)
    walls.extend(island_walls)
    
    # Bottom-right island
    island_walls = create_island(island_offset, -island_offset, island_size, island_size, wall_height)
    walls.extend(island_walls)
    
    # Define valid positions (in hallways, not in islands or walls)
    valid_positions = []
    
    # Horizontal hallway (middle)
    for x in range(-int(half_size) + 1, int(half_size)):
        valid_positions.append([x, 0])
    
    # Vertical hallway (middle)
    for y in range(-int(half_size) + 1, int(half_size)):
        valid_positions.append([0, y])
    
    # Top horizontal hallway
    for x in range(-int(half_size) + 1, int(half_size)):
        valid_positions.append([x, island_offset + island_size/2 + hallway_width/2])
    
    # Bottom horizontal hallway
    for x in range(-int(half_size) + 1, int(half_size)):
        valid_positions.append([x, -(island_offset + island_size/2 + hallway_width/2)])
    
    # Left vertical hallway
    for y in range(-int(half_size) + 1, int(half_size)):
        valid_positions.append([-(island_offset + island_size/2 + hallway_width/2), y])
    
    # Right vertical hallway
    for y in range(-int(half_size) + 1, int(half_size)):
        valid_positions.append([island_offset + island_size/2 + hallway_width/2, y])
    
    return {
        'walls': walls,
        'valid_positions': valid_positions,
        'env_size': env_size,
        'hallway_width': hallway_width
    }


def get_random_hallway_position(valid_positions):
    """Get a random valid position in the hallways."""
    pos = random.choice(valid_positions)
    return [pos[0], pos[1], 0.2]


def main():
    """Test the hallway environment."""
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    
    # Set up environment
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load floor
    plane_id = p.loadURDF("plane.urdf")
    
    # Create hallway environment
    print("Creating hallway environment...")
    env_data = create_hallway_environment(env_size=20, hallway_width=3)
    print(f"Created {len(env_data['walls'])} wall segments")
    print(f"Valid positions: {len(env_data['valid_positions'])}")
    
    # Load robot at a random valid position
    start_pos = get_random_hallway_position(env_data['valid_positions'])
    robot_orientation = p.getQuaternionFromEuler([0, 0, random.uniform(0, 2*math.pi)])
    robot_id = p.loadURDF("husky/husky.urdf", start_pos, robot_orientation)
    print(f"Robot placed at: {start_pos[:2]}")
    
    # Set camera view
    p.resetDebugVisualizerCamera(
        cameraDistance=25,
        cameraYaw=0,
        cameraPitch=-89,
        cameraTargetPosition=[0, 0, 0]
    )
    
    print("\n=== HALLWAY ENVIRONMENT ===")
    print("You should see a + shaped hallway pattern")
    print("4 gray islands with hallways between them")
    print("Robot (Husky) placed in one of the hallways")
    print("\nUse mouse to rotate view")
    print("Close window to exit")
    print("===========================\n")
    
    # Run simulation
    while p.isConnected():
        p.stepSimulation()
        time.sleep(1/240)
    
    print("Simulation ended.")


if __name__ == "__main__":
    main()
