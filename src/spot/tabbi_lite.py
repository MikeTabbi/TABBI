"""
TABBI Lite - Main Integration
Autonomous building exploration and navigation for Boston Dynamics Spot.
"""

import time
import cv2
import numpy as np
from typing import Optional, List, Dict

from bosdyn.client import create_standard_sdk
from bosdyn.client.robot import Robot
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_command import RobotCommandClient, blocking_stand, RobotCommandBuilder
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.api import image_pb2

from ultralytics import YOLO

from src.graph.graph_manager import GraphManager
from src.exploration.building_explorer import BuildingExplorer
from src.vision.sign_reader import SignReader


class TABBILite:
    """Main controller for TABBI Lite system."""
    
    def __init__(self, robot_ip: str, username: str, password: str):
        self.robot_ip = robot_ip
        self.username = username
        self.password = password
        
        # Components
        self.robot: Optional[Robot] = None
        self.graph = GraphManager()
        self.explorer: Optional[BuildingExplorer] = None
        self.sign_reader = SignReader()
        self.door_detector = YOLO('models/door_detector.pt')
        
        # State
        self.current_room: Optional[str] = None
        self.hallway_count = 0
        self.is_connected = False
        self._detected_signs = []
        
    def connect(self) -> bool:
        """Connect to Spot and initialize clients."""
        try:
            sdk = create_standard_sdk('TABBILite')
            self.robot = sdk.create_robot(self.robot_ip)
            self.robot.authenticate(self.username, self.password)
            self.robot.time_sync.wait_for_sync()
            
            # Initialize clients
            self.lease_client = self.robot.ensure_client(LeaseClient.default_service_name)
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            self.command_client = self.robot.ensure_client(RobotCommandClient.default_service_name)
            self.state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
            
            self.is_connected = True
            print("Connected to Spot successfully.")
            return True
            
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def power_on(self) -> bool:
        """Power on Spot and stand up."""
        try:
            self.lease = self.lease_client.acquire()
            self.lease_keepalive = LeaseKeepAlive(self.lease_client)
            
            self.robot.power_on(timeout_sec=20)
            blocking_stand(self.command_client, timeout_sec=10)
            
            print("Spot is powered on and standing.")
            return True
            
        except Exception as e:
            print(f"Power on failed: {e}")
            return False
    
    def power_off(self) -> None:
        """Safely power off Spot."""
        try:
            self.robot.power_off(cut_immediately=False)
            self.lease_keepalive.shutdown()
            print("Spot powered off safely.")
        except Exception as e:
            print(f"Power off error: {e}")
    
    def capture_image(self) -> Optional[np.ndarray]:
        """Capture image from Spot's hand camera."""
        try:
            image_responses = self.image_client.get_image_from_sources(['hand_color_image'])
            
            if not image_responses:
                return None
            
            image = image_responses[0]
            
            # Decode JPEG/compressed image
            img = cv2.imdecode(
                np.frombuffer(image.shot.image.data, dtype=np.uint8), 
                cv2.IMREAD_COLOR
            )
            
            if img is not None:
                print(f"Image shape: {img.shape}")
            
            return img
            
        except Exception as e:
            print(f"Image capture failed: {e}")
            return None
    
    def get_position(self) -> tuple:
        """Get Spot's current position from odometry."""
        try:
            state = self.state_client.get_robot_state()
            position = state.kinematic_state.transforms_snapshot.child_to_parent_edge_map['body'].parent_tform_child.position
            return (position.x, position.y, position.z)
        except Exception as e:
            print(f"Position error: {e}")
            return (0.0, 0.0, 0.0)
    
    def detect_doors(self, image: np.ndarray) -> List[Dict]:
        """Detect doors and signs in image using YOLO."""
        doors = []
        signs = []
        
        results = self.door_detector(image, verbose=False)
        
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                name = self.door_detector.names[cls]
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (int(x1), int(y1), int(x2), int(y2))
                
                if name == 'door' and conf > 0.3:
                    doors.append({
                        'bbox': bbox,
                        'center': ((x1 + x2) / 2, (y1 + y2) / 2),
                        'confidence': conf
                    })
                elif name == 'sign' and conf > 0.3:
                    signs.append({
                        'bbox': bbox,
                        'confidence': conf
                    })
        
        # Store signs for later use
        self._detected_signs = signs
        
        return doors
    
    def read_room_sign(self, image: np.ndarray, bbox: tuple) -> Optional[str]:
        """Read room sign using OCR."""
        x1, y1, x2, y2 = bbox
        
        # Expand bbox slightly to capture sign near door
        padding = 50
        h, w = image.shape[:2]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        
        cropped = image[y1:y2, x1:x2]
        return self.sign_reader.read_sign(cropped)
    
    def generate_room_id(self) -> str:
        """Generate a room ID when no sign is found."""
        self.hallway_count += 1
        return f"Hallway_{self.hallway_count}"
    
def scan_room(self) -> List[Dict]:
    """Scan current room for doors and identify rooms."""
    image = self.capture_image()
    
    if image is None:
        return []
    
    doors = self.detect_doors(image)
    discovered = []
    
    # First, try to read any detected signs
    sign_text = None
    if self._detected_signs:
        for sign in self._detected_signs:
            sign_text = self.sign_reader.read_sign_from_bbox(image, sign['bbox'])
            if sign_text:
                print(f"Sign detected (YOLO): {sign_text}")
                break
    
    # If no sign detected by YOLO, try OCR on full image
    if sign_text is None:
        sign_text = self.sign_reader.read_sign(image)
        if sign_text:
            print(f"Sign detected (full image OCR): {sign_text}")
    
    for door in doors:
        # Use sign text if found, otherwise auto-generate
        if sign_text:
            room_id = sign_text
        else:
            room_id = self.generate_room_id()
        
        position = self.get_position()
        
        discovered.append({
            'room_id': room_id,
            'position': position,
            'door_bbox': door['bbox'],
            'confidence': door['confidence']
        })
        
        print(f"Discovered: {room_id} (confidence: {door['confidence']:.2f})")
    
    return discovered
    
    def start_exploration(self, start_room: str = "Start_Room") -> None:
        """Begin autonomous exploration."""
        print(f"\n=== Starting Exploration from {start_room} ===\n")
        
        # Initialize
        self.current_room = start_room
        position = self.get_position()
        self.graph.add_room(start_room, position)
        
        self.explorer = BuildingExplorer(self.graph)
        self.explorer.start_exploration(start_room)
        
        exploration_count = 0
        max_explorations = 20  # Safety limit
        
        while exploration_count < max_explorations:
            print(f"\n--- Scanning {self.current_room} ---")
            
            # Scan for doors
            discovered = self.scan_room()
            
            # Add discovered rooms to graph
            for room_info in discovered:
                room_id = room_info['room_id']
                room_pos = room_info['position']
                
                if room_id not in self.graph.rooms:
                    self.graph.add_room(room_id, room_pos)
                    self.graph.add_door(self.current_room, room_id, room_pos)
                    print(f"Added to graph: {self.current_room} <-> {room_id}")
            
            # Mark current room explored
            self.graph.mark_explored(self.current_room)
            
            # Get next room from DFS
            next_room = self.explorer.get_next_room()
            
            if next_room is None:
                print("\n=== Exploration Complete ===")
                break
            
            print(f"Moving to: {next_room}")
            
            # TODO: Implement actual movement to next room
            # For now, simulate movement
            self.current_room = next_room
            self.explorer.move_to_room(next_room)
            
            exploration_count += 1
            time.sleep(2)  # Pause between scans
        
        # Save graph
        self.graph.save("building_graph.json")
        print(f"\nGraph saved. Total rooms: {len(self.graph.rooms)}")
    
    def navigate_to(self, target_room: str) -> bool:
        """Navigate to a specific room using A*."""
        print(f"\n=== Navigating to {target_room} ===\n")
        
        if target_room not in self.graph.rooms:
            print(f"Error: {target_room} not found in graph.")
            return False
        
        # Find path using A*
        path = self.graph.find_path(self.current_room, target_room)
        
        if not path:
            print(f"Error: No path found to {target_room}")
            return False
        
        print(f"Path: {' -> '.join(path)}")
        
        # Follow path
        for i, room in enumerate(path[1:], 1):
            print(f"Step {i}: Moving to {room}")
            
            # TODO: Implement actual movement
            # For now, simulate
            self.current_room = room
            time.sleep(1)
        
        print(f"\nArrived at {target_room}")
        return True


def main():
    """Main entry point."""
    # Configuration
    ROBOT_IP = "192.168.80.3"
    USERNAME = "admin"
    PASSWORD = "your_password_here"  # Replace with actual password
    
    # Initialize TABBI Lite
    tabbi = TABBILite(ROBOT_IP, USERNAME, PASSWORD)
    
    # Connect to Spot
    if not tabbi.connect():
        return
    
    try:
        # Power on
        if not tabbi.power_on():
            return
        
        # Start exploration
        tabbi.start_exploration("Room_101")
        
        # After exploration, test navigation
        print("\n" + "="*50)
        print("Exploration complete. Testing navigation...")
        print("="*50)
        
        # Example navigation command
        # tabbi.navigate_to("Room_301")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    finally:
        tabbi.power_off()


if __name__ == "__main__":
    main()