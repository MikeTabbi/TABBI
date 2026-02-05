import bosdyn.client
import bosdyn.client.util
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder, blocking_stand
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
import time

# CHANGE THESE:
SPOT_IP = "192.168.80.3"
USERNAME = "admin"
PASSWORD = "rzgnthrijszf"

def test_movement():
    """Test basic Spot movement."""
    
    print("=" * 50)
    print("SPOT MOVEMENT TEST")
    print("=" * 50)
    
    # Create SDK and connect
    sdk = bosdyn.client.create_standard_sdk('SpotMovementTest')
    robot = sdk.create_robot(SPOT_IP)
    bosdyn.client.util.authenticate(robot)
    
    # Get clients
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    
    # Acquire lease (with force take if needed)
    print("\n1. Acquiring lease (exclusive control)...")
    try:
        lease = lease_client.acquire()
    except bosdyn.client.lease.ResourceAlreadyClaimedError:
        print("   ⚠️  Lease already claimed (controller?), taking forcefully...")
        lease = lease_client.take()
    
    lease_keep_alive = LeaseKeepAlive(lease_client)
    print("   ✓ Lease acquired!")
    
    # Power on motors
    print("2. Powering on motors...")
    robot.power_on(timeout_sec=20)
    print("   ✓ Motors powered!")
    
    # Stand up
    print("3. Standing up...")
    blocking_stand(command_client, timeout_sec=10)
    print("   ✓ Standing!")
    
    print("\n" + "=" * 50)
    print("MOVEMENT TEST:")
    print("=" * 50)
    
    print("\n" + "=" * 50)
    print("MOVEMENT TEST:")
    print("=" * 50)
    
    # Import time for end_time calculation
    import time as time_module
    
    # Test 1: Move forward
    print("\n[Test 1] Moving forward 0.5 meters...")
    cmd = RobotCommandBuilder.synchro_velocity_command(
        v_x=0.5,  # Forward velocity (m/s)
        v_y=0.0,  # Left/right
        v_rot=0.0 # Rotation
    )
    end_time = time_module.time() + 2.0  # Command valid for 2 seconds
    command_client.robot_command(cmd, end_time_secs=end_time)
    time.sleep(2)
    print("   ✓ Forward complete!")
    
    # Stop
    print("[Stop] Stopping...")
    cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, 0)
    end_time = time_module.time() + 0.5
    command_client.robot_command(cmd, end_time_secs=end_time)
    time.sleep(1)
    
    # Test 2: Turn left
    print("\n[Test 2] Turning left...")
    cmd = RobotCommandBuilder.synchro_velocity_command(
        v_x=0.0,
        v_y=0.0,
        v_rot=0.5  # Positive = left, negative = right
    )
    end_time = time_module.time() + 2.0
    command_client.robot_command(cmd, end_time_secs=end_time)
    time.sleep(2)
    print("   ✓ Turn complete!")
    
    # Stop
    cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, 0)
    end_time = time_module.time() + 0.5
    command_client.robot_command(cmd, end_time_secs=end_time)
    time.sleep(1)
    
    # Test 3: Turn right
    print("\n[Test 3] Turning right...")
    cmd = RobotCommandBuilder.synchro_velocity_command(
        v_x=0.0,
        v_y=0.0,
        v_rot=-0.5
    )
    end_time = time_module.time() + 2.0
    command_client.robot_command(cmd, end_time_secs=end_time)
    time.sleep(2)
    print("   ✓ Turn complete!")
    
    # Stop
    cmd = RobotCommandBuilder.synchro_velocity_command(0, 0, 0)
    end_time = time_module.time() + 0.5
    command_client.robot_command(cmd, end_time_secs=end_time)
    
    # Sit down
    print("\n4. Sitting down...")
    cmd = RobotCommandBuilder.synchro_sit_command()
    command_client.robot_command(cmd)
    time.sleep(3)
    print("   ✓ Sitting!")
    
    # Power off
    print("5. Powering off...")
    robot.power_off(cut_immediately=False)
    print("   ✓ Powered off!")
    
    # Return lease
    lease_client.return_lease(lease)
    lease_keep_alive.shutdown()
    
    print("\n" + "=" * 50)
    print("MOVEMENT TEST SUCCESSFUL!")
    print("=" * 50)

if __name__ == "__main__":
    try:
        test_movement()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        