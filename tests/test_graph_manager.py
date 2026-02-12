# Graph Manager Test File
import os
import pytest
from src.graph.graph_manager import GraphManager, Room, Door


class TestRoomManagement:
    # room testing
    
    def test_add_room(self):
        gm = GraphManager()
        room = gm.add_room("Room_101", (0.0, 0.0, 0.0))
        
        assert room.id == "Room_101"
        assert room.position == (0.0, 0.0, 0.0)
        assert room.explored == False
        assert room.neighbors == []
    
    def test_add_duplicate_room(self):
        gm = GraphManager()
        room1 = gm.add_room("Room_101", (0.0, 0.0, 0.0))
        
        # different position
        room2 = gm.add_room("Room_101", (1.0, 1.0, 1.0)) 
        
        # returns a room that is there, not a new one
        assert room1 is room2
        assert room1.position == (0.0, 0.0, 0.0)
    
    def test_get_room(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        
        room = gm.get_room("Room_101")
        assert room is not None
        assert room.id == "Room_101"
        
        # a room (not real)
        assert gm.get_room("Room_999") is None
    
    def test_mark_explored(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        
        assert gm.rooms["Room_101"].explored == False
        gm.mark_explored("Room_101")
        assert gm.rooms["Room_101"].explored == True
    
    def test_get_unexplored(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Room_102", (1.0, 0.0, 0.0))
        gm.add_room("Room_103", (2.0, 0.0, 0.0))
        
        assert len(gm.get_unexplored()) == 3
        
        gm.mark_explored("Room_101")
        unexplored = gm.get_unexplored()
        
        assert len(unexplored) == 2
        assert "Room_101" not in unexplored
        assert "Room_102" in unexplored
        assert "Room_103" in unexplored


class TestDoorManagement:
    # door tests
    
    def test_add_door(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        
        door = gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0), "open")
        
        assert door.connects == ("Room_101", "Hallway_A")
        assert door.position == (0.5, 0.0, 0.0)
        assert door.state == "open"
    
    def test_door_updates_neighbors(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0))
        
        assert "Hallway_A" in gm.rooms["Room_101"].neighbors
        assert "Room_101" in gm.rooms["Hallway_A"].neighbors
    
    def test_get_door(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0))
        
        # looking for door
        door1 = gm.get_door("Room_101", "Hallway_A")
        door2 = gm.get_door("Hallway_A", "Room_101")
        
        assert door1 is not None
        assert door1 is door2
    
    def test_update_door_state(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0), "unknown")
        
        gm.update_door_state("Room_101", "Hallway_A", "open")
        
        door = gm.get_door("Room_101", "Hallway_A")
        assert door.state == "open"


class TestNavigation:
    # pathfinding test
    
    def test_get_neighbors(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        gm.add_room("Room_102", (2.0, 0.0, 0.0))
        
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0))
        gm.add_door("Hallway_A", "Room_102", (1.5, 0.0, 0.0))
        
        neighbors = gm.get_neighbors("Hallway_A")
        
        assert len(neighbors) == 2
        assert "Room_101" in neighbors
        assert "Room_102" in neighbors
    
    def test_find_path_simple(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0))
        
        path = gm.find_path("Room_101", "Hallway_A")
        
        assert path == ["Room_101", "Hallway_A"]
    
    def test_find_path_multiple_rooms(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        gm.add_room("Room_102", (2.0, 0.0, 0.0))
        gm.add_room("Room_103", (2.0, 1.0, 0.0))
        
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0))
        gm.add_door("Hallway_A", "Room_102", (1.5, 0.0, 0.0))
        gm.add_door("Hallway_A", "Room_103", (1.0, 0.5, 0.0))
        
        path = gm.find_path("Room_101", "Room_103")
        
        assert path == ["Room_101", "Hallway_A", "Room_103"]
    
    def test_find_path_same_room(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        
        path = gm.find_path("Room_101", "Room_101")
        
        assert path == ["Room_101"]
    
    def test_find_path_no_path(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Room_102", (1.0, 0.0, 0.0))
        # no door connecting them
        
        path = gm.find_path("Room_101", "Room_102")
        
        assert path == []
    
    def test_find_path_avoids_inaccessible(self):
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        gm.add_room("Room_102", (2.0, 0.0, 0.0))
        gm.add_room("Hallway_B", (1.0, 1.0, 0.0))
        
        # path blocked
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0), "inaccessible")
        gm.add_door("Hallway_A", "Room_102", (1.5, 0.0, 0.0))
        
        # new path available
        gm.add_door("Room_101", "Hallway_B", (0.5, 0.5, 0.0), "open")
        gm.add_door("Hallway_B", "Room_102", (1.5, 0.5, 0.0), "open")
        
        path = gm.find_path("Room_101", "Room_102")
        
        assert path == ["Room_101", "Hallway_B", "Room_102"]
        assert "Hallway_A" not in path


class TestPersistence:
    # saving and loading
    
    def test_save_and_load(self, tmp_path):
        # create graph
        gm = GraphManager()
        gm.add_room("Room_101", (0.0, 0.0, 0.0))
        gm.add_room("Hallway_A", (1.0, 0.0, 0.0))
        gm.add_door("Room_101", "Hallway_A", (0.5, 0.0, 0.0), "open")
        gm.mark_explored("Room_101")
        
        # save
        filepath = tmp_path / "test_graph.json"
        gm.save(str(filepath))
        
        # load into new graph
        gm2 = GraphManager()
        gm2.load(str(filepath))
        
        # verification
        assert len(gm2.rooms) == 2
        assert len(gm2.doors) == 1
        assert gm2.rooms["Room_101"].explored == True
        assert gm2.rooms["Hallway_A"].explored == False
        assert "Hallway_A" in gm2.rooms["Room_101"].neighbors
        assert gm2.doors[0].state == "open"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])