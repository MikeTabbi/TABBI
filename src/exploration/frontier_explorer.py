import numpy as np
import math


# Cell states
UNKNOWN = -1
FREE = 0
OCCUPIED = 1


class FrontierExplorer:
    """
    Manages a 2D grid map and finds frontiers for exploration.
    """
    
    def __init__(self, map_size=10.0, resolution=0.2):
        """
        Initialize the explorer with an empty map.
        
        Args:
            map_size: Size of map in meters (map is square)
            resolution: Meters per grid cell
        """
        self.map_size = map_size
        self.resolution = resolution
        
        # Calculate grid dimensions
        self.grid_width = int(map_size / resolution)
        self.grid_height = int(map_size / resolution)
        
        # Initialize grid as all UNKNOWN
        self.grid = np.full((self.grid_height, self.grid_width), UNKNOWN, dtype=np.int8)
        
        # Track exploration progress
        self.total_cells = self.grid_width * self.grid_height
        
        print(f"FrontierExplorer initialized:")
        print(f"  Map size: {map_size}m x {map_size}m")
        print(f"  Resolution: {resolution}m per cell")
        print(f"  Grid size: {self.grid_width} x {self.grid_height} cells")
    
    def meters_to_cell(self, x, y):
        """Convert position in meters to grid cell coordinates."""
        cell_x = int((x + self.map_size / 2) / self.resolution)
        cell_y = int((y + self.map_size / 2) / self.resolution)
        
        # Clamp to grid bounds
        cell_x = max(0, min(cell_x, self.grid_width - 1))
        cell_y = max(0, min(cell_y, self.grid_height - 1))
        
        return cell_x, cell_y
    
    def cell_to_meters(self, cell_x, cell_y):
        """Convert grid cell coordinates to position in meters."""
        x = (cell_x * self.resolution) - (self.map_size / 2) + (self.resolution / 2)
        y = (cell_y * self.resolution) - (self.map_size / 2) + (self.resolution / 2)
        return x, y
    
    def update_map(self, robot_x, robot_y, sensor_range=2.0, walls=None):
        """
        Update the map based on robot's current position and sensor readings.
        
        Args:
            robot_x, robot_y: Robot position in meters
            sensor_range: How far the robot can see in meters
            walls: List of wall positions [(x1,y1,x2,y2), ...] for collision checking
        """
        robot_cell_x, robot_cell_y = self.meters_to_cell(robot_x, robot_y)
        
        # Calculate range in cells
        range_cells = int(sensor_range / self.resolution)
        
        # Update cells within sensor range
        for dy in range(-range_cells, range_cells + 1):
            for dx in range(-range_cells, range_cells + 1):
                cell_x = robot_cell_x + dx
                cell_y = robot_cell_y + dy
                
                # Check bounds
                if not (0 <= cell_x < self.grid_width and 0 <= cell_y < self.grid_height):
                    continue
                
                # Check distance
                distance = math.sqrt(dx**2 + dy**2) * self.resolution
                if distance > sensor_range:
                    continue
                
                # Get cell center in meters
                cell_center_x, cell_center_y = self.cell_to_meters(cell_x, cell_y)
                
                # Check if this cell is a wall
                is_wall = self._check_wall(cell_center_x, cell_center_y, walls)
                
                if is_wall:
                    self.grid[cell_y, cell_x] = OCCUPIED
                else:
                    self.grid[cell_y, cell_x] = FREE
    
    def _check_wall(self, x, y, walls):
        """Check if a position is inside a wall."""
        if walls is None:
            # Simple boundary check for our 10x10 room
            margin = self.map_size / 2 - 0.3  # Wall thickness margin
            if abs(x) > margin or abs(y) > margin:
                return True
            return False
        
        # Check against provided wall positions
        for wall in walls:
            # Wall format depends on implementation
            pass
        
        return False
    
    def find_frontiers(self):
        """
        Find all frontier cells.
        A frontier is a FREE cell adjacent to an UNKNOWN cell.
        
        Returns:
            List of (cell_x, cell_y) tuples
        """
        frontiers = []
        
        # Check each cell
        for y in range(self.grid_height):
            for x in range(self.grid_width):
                # Only FREE cells can be frontiers
                if self.grid[y, x] != FREE:
                    continue
                
                # Check if any neighbor is UNKNOWN
                is_frontier = False
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    
                    # Check bounds
                    if not (0 <= nx < self.grid_width and 0 <= ny < self.grid_height):
                        continue
                    
                    if self.grid[ny, nx] == UNKNOWN:
                        is_frontier = True
                        break
                
                if is_frontier:
                    frontiers.append((x, y))
        
        return frontiers
    
    def get_nearest_frontier(self, robot_x, robot_y):
        """
        Find the closest frontier to the robot.
        
        Args:
            robot_x, robot_y: Robot position in meters
            
        Returns:
            (x, y) in meters, or None if no frontiers exist
        """
        frontiers = self.find_frontiers()
        
        if not frontiers:
            return None  # Exploration complete!
        
        robot_cell_x, robot_cell_y = self.meters_to_cell(robot_x, robot_y)
        
        nearest = None
        min_distance = float('inf')
        
        for frontier_x, frontier_y in frontiers:
            distance = math.sqrt((frontier_x - robot_cell_x)**2 + (frontier_y - robot_cell_y)**2)
            if distance < min_distance:
                min_distance = distance
                nearest = (frontier_x, frontier_y)
        
        if nearest:
            return self.cell_to_meters(nearest[0], nearest[1])
        return None
    
    def get_exploration_progress(self):
        """
        Calculate what percentage of the map has been explored.
        
        Returns:
            Float between 0 and 1
        """
        explored = np.sum(self.grid != UNKNOWN)
        return explored / self.total_cells
    
    def is_exploration_complete(self):
        """Check if there are no more frontiers to explore."""
        return len(self.find_frontiers()) == 0
    
    def get_map_display(self):
        """
        Get a string representation of the map for display.
        
        Returns:
            String showing the map
        """
        symbols = {UNKNOWN: '?', FREE: '.', OCCUPIED: '#'}
        lines = []
        
        for y in range(self.grid_height - 1, -1, -1):  # Top to bottom
            line = ''
            for x in range(self.grid_width):
                line += symbols[self.grid[y, x]]
            lines.append(line)
        
        return '\n'.join(lines)


# Test the frontier explorer
if __name__ == "__main__":
    print("Testing FrontierExplorer...")
    print()
    
    # Create explorer
    explorer = FrontierExplorer(map_size=10.0, resolution=0.5)
    
    # Simulate robot at center
    robot_x, robot_y = 0.0, 0.0
    
    print(f"Robot at ({robot_x}, {robot_y})")
    print(f"Initial exploration: {explorer.get_exploration_progress()*100:.1f}%")
    print()
    
    # Update map with robot's sensor
    explorer.update_map(robot_x, robot_y, sensor_range=2.0)
    
    print(f"After first scan: {explorer.get_exploration_progress()*100:.1f}%")
    
    # Find frontiers
    frontiers = explorer.find_frontiers()
    print(f"Frontiers found: {len(frontiers)}")
    
    # Get nearest frontier
    nearest = explorer.get_nearest_frontier(robot_x, robot_y)
    print(f"Nearest frontier: {nearest}")
    print()
    
    # Show map
    print("Map (? = unknown, . = free, # = wall):")
    print(explorer.get_map_display())
    print()
    
    # Simulate moving to a few positions
    positions = [(2, 0), (2, 2), (0, 2), (-2, 2), (-2, 0)]
    
    for px, py in positions:
        explorer.update_map(px, py, sensor_range=2.0)
        progress = explorer.get_exploration_progress() * 100
        frontiers = len(explorer.find_frontiers())
        print(f"Robot at ({px}, {py}): {progress:.1f}% explored, {frontiers} frontiers")
    
    print()
    print("Final map:")
    print(explorer.get_map_display())
    
    print()
    print(f"Exploration complete: {explorer.is_exploration_complete()}")
    