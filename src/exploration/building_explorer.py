"""
Building Explorer for TABBI Lite
Uses DFS to systematically explore all rooms.
"""

from typing import List, Optional, Callable
from src.graph.graph_manager import GraphManager


class BuildingExplorer:
    """Explores building using DFS algorithm."""
    
    def __init__(self, graph: GraphManager):
        self.graph = graph
        self.exploration_stack: List[str] = []
        self.current_room: Optional[str] = None
        self.visited: set = set()
    
    def start_exploration(self, start_room: str) -> None:
        """Initialize exploration from starting room."""
        self.current_room = start_room
        self.exploration_stack = [start_room]
        self.visited = {start_room}
        
        if start_room not in self.graph.rooms:
            self.graph.add_room(start_room, (0.0, 0.0, 0.0))
    
    def get_next_room(self) -> Optional[str]:
        """
        Get next room to explore using DFS.
        Returns None if exploration complete.
        """
        while self.exploration_stack:
            room = self.exploration_stack[-1]
            
            # Find unvisited neighbor
            for neighbor in self.graph.get_neighbors(room):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    return neighbor
            
            # No unvisited neighbors, backtrack
            self.exploration_stack.pop()
            self.graph.mark_explored(room)
        
        return None
    
    def move_to_room(self, room_id: str) -> None:
        """Update current position after moving."""
        self.current_room = room_id
        self.exploration_stack.append(room_id)
    
    def add_discovered_room(self, room_id: str, position: tuple, door_position: tuple) -> None:
        """Add newly discovered room and door."""
        self.graph.add_room(room_id, position)
        self.graph.add_door(self.current_room, room_id, door_position)
    
    def is_complete(self) -> bool:
        """Check if exploration is complete."""
        return len(self.exploration_stack) == 0
    
    def explore(
        self,
        scan_room: Callable[[], List[dict]],
        move_to: Callable[[str], bool],
        get_position: Callable[[], tuple]
    ) -> None:
        """
        Main exploration loop.
        
        Args:
            scan_room: Function that returns list of detected doors with room_id and position
            move_to: Function that moves robot to a room, returns success
            get_position: Function that returns current (x, y, z) position
        """
        while not self.is_complete():
            # Scan current room for doors
            doors = scan_room()
            
            for door in doors:
                room_id = door['room_id']
                door_pos = door['position']
                
                if room_id not in self.graph.rooms:
                    self.add_discovered_room(room_id, door_pos, door_pos)
            
            self.graph.mark_explored(self.current_room)
            
            # Get next room to visit
            next_room = self.get_next_room()
            
            if next_room is None:
                break
            
            # Move to next room
            if move_to(next_room):
                self.move_to_room(next_room)


if __name__ == "__main__":
    gm = GraphManager()
    explorer = BuildingExplorer(gm)
    
    gm.add_room("Room_101", (0, 0, 0))
    gm.add_room("Hallway_A", (1, 0, 0))
    gm.add_room("Room_102", (2, 0, 0))
    gm.add_door("Room_101", "Hallway_A", (0.5, 0, 0))
    gm.add_door("Hallway_A", "Room_102", (1.5, 0, 0))
    
    explorer.start_exploration("Room_101")
    
    while True:
        next_room = explorer.get_next_room()
        if next_room is None:
            break
        print(f"Moving to: {next_room}")
        explorer.move_to_room(next_room)
    
    print(f"Exploration complete. Rooms: {list(gm.rooms.keys())}")