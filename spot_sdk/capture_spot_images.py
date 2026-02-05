import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.api import image_pb2
import cv2
import numpy as np
import os
from datetime import datetime

# CHANGE THESE:
SPOT_IP = "192.168.80.3"
USERNAME = "admin"
PASSWORD = "your_password"
OUTPUT_DIR = "./spot_images"

def capture_images(num_images=50):
    """Capture images from Spot's cameras."""
    
    print("=" * 50)
    print("SPOT IMAGE CAPTURE")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nImages will be saved to: {OUTPUT_DIR}")
    
    # Connect
    sdk = bosdyn.client.create_standard_sdk('SpotImageCapture')
    robot = sdk.create_robot(SPOT_IP)
    bosdyn.client.util.authenticate(robot)
    
    # Get image client
    image_client = robot.ensure_client(ImageClient.default_service_name)
    
    # Available cameras
    cameras = [
        'frontleft_fisheye_image',
        'frontright_fisheye_image',
        'left_fisheye_image',
        'right_fisheye_image',
        'back_fisheye_image'
    ]
    
    print(f"\nCapturing {num_images} images from each camera...")
    print("Move Spot around to get different angles!\n")
    
    for i in range(num_images):
        print(f"Image {i+1}/{num_images}...", end=" ")
        
        # Request images from all cameras
        image_requests = [
            image_pb2.ImageRequest(
                image_source_name=camera,
                quality_percent=75
            ) for camera in cameras
        ]
        
        # Get images
        image_responses = image_client.get_image(image_requests)
        
        # Save each camera image
        for j, response in enumerate(image_responses):
            if response.shot.image.format == image_pb2.Image.FORMAT_RAW:
                # Decode image
                img = np.frombuffer(response.shot.image.data, dtype=np.uint8)
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                
                # Save
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{OUTPUT_DIR}/spot_{cameras[j]}_{timestamp}_{i:03d}.jpg"
                cv2.imwrite(filename, img)
        
        print("✓")
        
        # Wait before next capture (gives you time to move Spot)
        if i < num_images - 1:
            import time
            time.sleep(2)
    
    print("\n" + "=" * 50)
    print(f"CAPTURED {num_images * len(cameras)} IMAGES!")
    print(f"Saved to: {OUTPUT_DIR}")
    print("=" * 50)

if __name__ == "__main__":
    try:
        # Capture 50 images (you can change this number)
        capture_images(num_images=50)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()