# Graph Manager
import json
import math
import heapq
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional


@dataclass
class Room:
    """Represents a room (node) in the building graph."""
    id: str
    position: Tuple[float, float, float]  # (x, y, z) from Spot odometry
    explored: bool = False
    neighbors: List[str] = field(default_factory=list)


@dataclass
class Door:
    """Represents a door (edge) connecting two rooms."""
    connects: Tuple[str, str]  # (room1_id, room2_id)
    position: Tuple[float, float, float]
    state: str = "unknown"  # "open", "closed", "inaccessible", "unknown"


class GraphManager:
    """
    Manages the building graph for TABBI Lite.
    
    Rooms are nodes, doors are edges.
    Supports DFS exploration and A* navigation.
    """
    
    def __init__(self):
        self.rooms: Dict[str, Room] = {}
        self.doors: List[Door] = []
    
    # ==================== Room Management ====================
    
    def add_room(self, room_id: str, position: Tuple[float, float, float]) -> Room:
        """Add a room to the graph."""
        if room_id not in self.rooms:
            self.rooms[room_id] = Room(id=room_id, position=position)
        return self.rooms[room_id]
    
    def get_room(self, room_id: str) -> Optional[Room]:
        """Get a room by ID."""
        return self.rooms.get(room_id)
    
    def mark_explored(self, room_id: str) -> None:
        """Mark a room as fully explored."""
        if room_id in self.rooms:
            self.rooms[room_id].explored = True
    
    def get_unexplored(self) -> List[str]:
        """Return list of unexplored room IDs."""
        return [room.id for room in self.rooms.values() if not room.explored]
    
    # ==================== Door Management ====================
    
    def add_door(
        self, 
        room1_id: str, 
        room2_id: str, 
        position: Tuple[float, float, float],
        state: str = "unknown"
    ) -> Door:
        """Add a door connecting two rooms."""
        # Create door
        door = Door(connects=(room1_id, room2_id), position=position, state=state)
        self.doors.append(door)
        
        # Update neighbors (bidirectional)
        if room1_id in self.rooms and room2_id not in self.rooms[room1_id].neighbors:
            self.rooms[room1_id].neighbors.append(room2_id)
        
        if room2_id in self.rooms and room1_id not in self.rooms[room2_id].neighbors:
            self.rooms[room2_id].neighbors.append(room1_id)
        
        return door
    
    def get_door(self, room1_id: str, room2_id: str) -> Optional[Door]:
        """Get door connecting two rooms."""
        for door in self.doors:
            if set(door.connects) == {room1_id, room2_id}:
                return door
        return None
    
    def update_door_state(self, room1_id: str, room2_id: str, state: str) -> None:
        """Update the state of a door."""
        door = self.get_door(room1_id, room2_id)
        if door:
            door.state = state
    
    # ==================== Navigation ====================
    
    def get_neighbors(self, room_id: str) -> List[str]:
        """Get all rooms connected to a room."""
        if room_id in self.rooms:
            return self.rooms[room_id].neighbors
        return []
    
    def _heuristic(self, room_id: str, goal_id: str) -> float:
        """Euclidean distance heuristic for A*."""
        pos1 = self.rooms[room_id].position
        pos2 = self.rooms[goal_id].position
        return math.sqrt(
            (pos2[0] - pos1[0]) ** 2 +
            (pos2[1] - pos1[1]) ** 2 +
            (pos2[2] - pos1[2]) ** 2
        )
    
    def find_path(self, start_id: str, goal_id: str) -> List[str]:
        """
        A* pathfinding from start to goal.
        Returns list of room IDs representing the path.
        """
        if start_id not in self.rooms or goal_id not in self.rooms:
            return []
        
        if start_id == goal_id:
            return [start_id]
        
        # Priority queue: (f_score, room_id)
        open_set = [(0, start_id)]
        came_from: Dict[str, str] = {}
        
        g_score: Dict[str, float] = {start_id: 0}
        f_score: Dict[str, float] = {start_id: self._heuristic(start_id, goal_id)}
        
        open_set_hash = {start_id}  # For O(1) membership check
        
        while open_set:
            # Get node with lowest f_score
            _, current = heapq.heappop(open_set)
            open_set_hash.discard(current)
            
            if current == goal_id:
                return self._reconstruct_path(came_from, current)
            
            for neighbor in self.get_neighbors(current):
                # Skip inaccessible doors
                door = self.get_door(current, neighbor)
                if door and door.state == "inaccessible":
                    continue
                
                tentative_g = g_score[current] + 1  # Cost = 1 per door
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_id)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return []  # No path found
    
    def _reconstruct_path(self, came_from: Dict[str, str], current: str) -> List[str]:
        """Reconstruct path from came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.insert(0, current)
        return path
    
    # ==================== Persistence ====================
    
    def save(self, filepath: str) -> None:
        """Save graph to JSON file."""
        data = {
            "rooms": {
                room_id: {
                    "id": room.id,
                    "position": room.position,
                    "explored": room.explored,
                    "neighbors": room.neighbors
                }
                for room_id, room in self.rooms.items()
            },
            "doors": [
                {
                    "connects": door.connects,
                    "position": door.position,
                    "state": door.state
                }
                for door in self.doors
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str) -> None:
        """Load graph from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Clear existing data
        self.rooms = {}
        self.doors = []
        
        # Load rooms
        for room_id, room_data in data["rooms"].items():
            room = Room(
                id=room_data["id"],
                position=tuple(room_data["position"]),
                explored=room_data["explored"],
                neighbors=room_data["neighbors"]
            )
            self.rooms[room_id] = room
        
        # Load doors
        for door_data in data["doors"]:
            door = Door(
                connects=tuple(door_data["connects"]),
                position=tuple(door_data["position"]),
                state=door_data["state"]
            )
            self.doors.append(door)
    
    def __repr__(self) -> str:
        return f"GraphManager(rooms={len(self.rooms)}, doors={len(self.doors)})"