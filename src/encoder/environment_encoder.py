import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os


class EnvironmentEncoder(nn.Module):
    """
    Encodes environment images into 128-dimensional fingerprints.
    Uses pre-trained ResNet-18 as feature extractor.
    """
    
    def __init__(self, fingerprint_size=128):
        super().__init__()
        
        self.fingerprint_size = fingerprint_size
        
        # Load pre-trained ResNet-18
        print("Loading pre-trained ResNet-18...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # Remove the final classification layer
        # ResNet-18 ends with: avgpool -> fc (512 -> 1000)
        # We keep everything except fc
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Freeze ResNet weights (we don't train them)
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        
        # Add our custom fingerprint layer: 512 -> 128
        self.fingerprint_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, fingerprint_size),
            nn.ReLU(),
            nn.Linear(fingerprint_size, fingerprint_size)
        )
        
        # Image preprocessing (what ResNet expects)
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]    # ImageNet stds
            )
        ])
        
        # Set to evaluation mode
        self.eval()
        
        print(f"EnvironmentEncoder initialized:")
        print(f"  Feature extractor: ResNet-18 (pre-trained)")
        print(f"  Fingerprint size: {fingerprint_size}")
    
    def encode(self, image):
        """
        Convert an image to a fingerprint.
        
        Args:
            image: PIL Image, numpy array, or torch tensor
            
        Returns:
            numpy array of shape (128,) - the fingerprint
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Preprocess
        processed = self.preprocess(image)
        
        # Add batch dimension: (3, 224, 224) -> (1, 3, 224, 224)
        batched = processed.unsqueeze(0)
        
        # Extract features (no gradient needed)
        with torch.no_grad():
            features = self.feature_extractor(batched)  # (1, 512, 1, 1)
            fingerprint = self.fingerprint_layer(features)  # (1, 128)
        
        # Normalize the fingerprint (unit length)
        fingerprint = fingerprint / fingerprint.norm()
        
        # Convert to numpy
        return fingerprint.squeeze().numpy()
    
    def encode_batch(self, images):
        """
        Encode multiple images at once.
        
        Args:
            images: List of images
            
        Returns:
            numpy array of shape (N, 128)
        """
        fingerprints = []
        for image in images:
            fingerprints.append(self.encode(image))
        return np.array(fingerprints)
    
    @staticmethod
    def compare(fingerprint1, fingerprint2):
        """
        Compare two fingerprints using cosine similarity.
        
        Args:
            fingerprint1, fingerprint2: numpy arrays of shape (128,)
            
        Returns:
            float between -1 and 1 (1 = identical)
        """
        # Cosine similarity
        dot_product = np.dot(fingerprint1, fingerprint2)
        norm1 = np.linalg.norm(fingerprint1)
        norm2 = np.linalg.norm(fingerprint2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    @staticmethod
    def save_fingerprint(fingerprint, filepath):
        """Save a fingerprint to a file."""
        np.save(filepath, fingerprint)
        print(f"Fingerprint saved to: {filepath}")
    
    @staticmethod
    def load_fingerprint(filepath):
        """Load a fingerprint from a file."""
        fingerprint = np.load(filepath)
        print(f"Fingerprint loaded from: {filepath}")
        return fingerprint


def create_test_image(color='random'):
    """Create a simple test image."""
    if color == 'random':
        # Random noise image
        data = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    elif color == 'red':
        data = np.zeros((224, 224, 3), dtype=np.uint8)
        data[:, :, 0] = 200  # Red channel
    elif color == 'blue':
        data = np.zeros((224, 224, 3), dtype=np.uint8)
        data[:, :, 2] = 200  # Blue channel
    elif color == 'hallway':
        # Simulate a hallway (gray with lines)
        data = np.full((224, 224, 3), 180, dtype=np.uint8)
        data[100:120, :, :] = 100  # Floor line
        data[:, 50:60, :] = 120    # Left wall
        data[:, 170:180, :] = 120  # Right wall
    elif color == 'room':
        # Simulate a room (different pattern)
        data = np.full((224, 224, 3), 200, dtype=np.uint8)
        data[150:224, :, :] = 100  # Floor
        data[:100, :, :] = 220     # Ceiling
    else:
        data = np.full((224, 224, 3), 128, dtype=np.uint8)
    
    return Image.fromarray(data)


# Test the encoder
if __name__ == "__main__":
    print("Testing EnvironmentEncoder...")
    print()
    
    # Create encoder
    encoder = EnvironmentEncoder(fingerprint_size=128)
    print()
    
    # Create test images
    print("Creating test images...")
    img_hallway1 = create_test_image('hallway')
    img_hallway2 = create_test_image('hallway')  # Same type
    img_room = create_test_image('room')          # Different type
    img_random = create_test_image('random')      # Very different
    
    # Encode images
    print("\nEncoding images...")
    fp_hallway1 = encoder.encode(img_hallway1)
    fp_hallway2 = encoder.encode(img_hallway2)
    fp_room = encoder.encode(img_room)
    fp_random = encoder.encode(img_random)
    
    print(f"Fingerprint shape: {fp_hallway1.shape}")
    print(f"Fingerprint sample: {fp_hallway1[:5]}...")
    
    # Compare fingerprints
    print("\nComparing fingerprints:")
    print(f"  Hallway1 vs Hallway2 (same type):  {encoder.compare(fp_hallway1, fp_hallway2):.4f}")
    print(f"  Hallway1 vs Room (different):      {encoder.compare(fp_hallway1, fp_room):.4f}")
    print(f"  Hallway1 vs Random (very diff):    {encoder.compare(fp_hallway1, fp_random):.4f}")
    print(f"  Hallway1 vs Hallway1 (identical):  {encoder.compare(fp_hallway1, fp_hallway1):.4f}")
    
    # Test save/load
    print("\nTesting save/load...")
    test_path = "test_fingerprint.npy"
    encoder.save_fingerprint(fp_hallway1, test_path)
    loaded_fp = encoder.load_fingerprint(test_path)
    print(f"  Original vs Loaded: {encoder.compare(fp_hallway1, loaded_fp):.4f}")
    
    # Clean up test file
    os.remove(test_path)
    print(f"  Test file removed.")
    
    print("\n" + "=" * 50)
    print("EnvironmentEncoder test complete!")
    print("=" * 50)
    