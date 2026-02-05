import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_state import RobotStateClient

# CHANGE THESE:
SPOT_IP = "192.168.80.3"  # <-- Change to your Spot's IP
USERNAME = "admin"         # <-- Change to your username
PASSWORD = "rzgnthrijszf" # <-- Change to your password

def test_connection():
    """Test basic connection to Spot."""
    
    print("=" * 50)
    print("SPOT CONNECTION TEST")
    print("=" * 50)
    
    # Create SDK
    sdk = bosdyn.client.create_standard_sdk('SpotConnectionTest')
    
    # Create robot instance
    print(f"\n1. Connecting to Spot at {SPOT_IP}...")
    robot = sdk.create_robot(SPOT_IP)
    
    # Authenticate
    print(f"2. Authenticating as {USERNAME}...")
    bosdyn.client.util.authenticate(robot)
    print("   ✓ Authentication successful!")
    
    # Check robot state
    print("3. Checking robot state...")
    state_client = robot.ensure_client(RobotStateClient.default_service_name)
    state = state_client.get_robot_state()
    
    print(f"   ✓ Robot ID: {state.robot_id}")
    print(f"   ✓ Battery: {state.power_state.locomotion_charge_percentage.value:.1f}%")
    print(f"   ✓ E-Stop: {state.estop_states}")
    
    print("\n" + "=" * 50)
    print("CONNECTION TEST SUCCESSFUL!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_connection()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nCommon issues:")
        print("- Wrong IP address")
        print("- Wrong username/password")
        print("- Not connected to Spot's WiFi")
        print("- Spot is powered off")