import easyocr
import cv2
import re
from typing import Optional, Tuple
import numpy as np


class SignReader:
    """Reads text from door signs using OCR."""
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)
    
    def read_sign(self, image: np.ndarray) -> Optional[str]:
        """
        Read text from an image.
        Returns cleaned room number or None.
        """
        results = self.reader.readtext(image)
        
        if not results:
            return None
        
        # Combine all detected text
        text = ' '.join([r[1] for r in results])
        
        # Clean and extract room number
        room_id = self._extract_room_number(text)
        
        return room_id
    
    def read_sign_from_bbox(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[str]:
        """
        Read text from a bounding box region.
        bbox: (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = bbox
        cropped = image[y1:y2, x1:x2]
        return self.read_sign(cropped)
    
    def _extract_room_number(self, text: str) -> Optional[str]:
        """Extract room number from OCR text."""
        text = text.upper().strip()
        
        # Pattern: "ROOM 301", "RM 301", "301", "ROOM-301"
        patterns = [
            r'ROOM\s*[-:]?\s*(\d+[A-Z]?)',
            r'RM\s*[-:]?\s*(\d+[A-Z]?)',
            r'^(\d{2,4}[A-Z]?)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return f"Room_{match.group(1)}"
        
        # If no pattern matched but has text, return cleaned text
        if text:
            clean = re.sub(r'[^A-Z0-9]', '_', text)
            return clean if clean else None
        
        return None


if __name__ == "__main__":
    import sys
    
    reader = SignReader()
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        image = cv2.imread(image_path)
        result = reader.read_sign(image)
        print(f"Detected: {result}")
    else:
        print("Usage: python sign_reader.py <image_path>")