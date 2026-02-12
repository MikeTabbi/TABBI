import os
import time
import cv2
import numpy as np
from datetime import datetime
from bosdyn.client import create_standard_sdk
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.api import image_pb2


class SpotImageCapture:
    """Captures and saves images from Spot's cameras."""
    
    # Available cameras on Spot
    CAMERAS = [
        'hand_color_image',  # Arm camera (best quality)
    ]
    
    def __init__(self, hostname: str, username: str = "admin", password: str = "your_password"):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.sdk = None
        self.robot = None
        self.image_client = None
    
    def connect(self) -> bool:
        """Connect to Spot."""
        try:
            self.sdk = create_standard_sdk('ImageCaptureClient')
            self.robot = self.sdk.create_robot(self.hostname)
            self.robot.authenticate(self.username, self.password)
            self.image_client = self.robot.ensure_client(ImageClient.default_service_name)
            print(f"Connected to Spot at {self.hostname}")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False
    
    def capture_image(self, camera: str = 'hand_color_image') -> np.ndarray:
        """Capture single image from specified camera."""
        image_request = [
            build_image_request(
                camera,
                quality_percent=100,
                image_format=image_pb2.Image.FORMAT_JPEG
            )
        ]
        
        image_responses = self.image_client.get_image(image_request)
        
        if image_responses:
            image_data = image_responses[0].shot.image.data
            np_arr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image
        
        return None
    
    def capture_all_cameras(self) -> dict:
        """Capture images from all cameras."""
        images = {}
        for camera in self.CAMERAS:
            try:
                img = self.capture_image(camera)
                if img is not None:
                    images[camera] = img
            except Exception as e:
                print(f"Failed to capture {camera}: {e}")
        return images
    
    def save_image(self, image: np.ndarray, folder: str, prefix: str = "door") -> str:
        """Save image with timestamp."""
        os.makedirs(folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, image)
        print(f"Saved: {filepath}")
        return filepath
    
    def capture_session(self, output_folder: str, prefix: str = "door", interval: float = 2.0):
        """
        Interactive capture session.
        Press SPACE to capture, Q to quit.
        """
        print("\n=== Image Capture Session ===")
        print("Controls:")
        print("  SPACE - Capture image")
        print("  Q     - Quit")
        print(f"\nSaving to: {output_folder}\n")
        
        os.makedirs(output_folder, exist_ok=True)
        count = 0
        
        while True:
            # Show live preview from arm camera
            img = self.capture_image('hand_color_image')
            if img is not None:
                # Resize for preview
                preview = cv2.resize(img, (640, 480))
                cv2.putText(preview, f"Captured: {count}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(preview, "SPACE=capture, Q=quit", (10, 460),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.imshow("Spot Arm Camera", preview)
            
            key = cv2.waitKey(100) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Capture from arm camera
                images = self.capture_all_cameras()
                for cam_name, cam_img in images.items():
                    cam_short = cam_name.replace('_image', '')
                    self.save_image(cam_img, output_folder, f"{prefix}_{cam_short}")
                count += 1
                print(f"Captured {count} images")
        
        cv2.destroyAllWindows()
        print(f"\nSession complete. Captured {count} images.")


def main():
    """Main capture function."""
    # Update these with your Spot's details
    SPOT_IP = "192.168.80.3"
    USERNAME = "admin"
    PASSWORD = "p0zv3tto3958"  # Change to your password
    
    OUTPUT_FOLDER = "data/door_images"
    
    capture = SpotImageCapture(SPOT_IP, USERNAME, PASSWORD)
    
    if capture.connect():
        capture.capture_session(OUTPUT_FOLDER, prefix="door")
    else:
        print("Could not connect to Spot")


if __name__ == "__main__":
    main()