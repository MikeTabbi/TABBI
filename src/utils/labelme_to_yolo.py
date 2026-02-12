# convert JSON to YOLO
import os
import json
import glob
from pathlib import Path


def convert_labelme_to_yolo(json_dir: str, output_dir: str = None):
    """
    Convert labelme JSON files to YOLO txt format.
    
    Args:
        json_dir: Directory containing labelme JSON files
        output_dir: Output directory for YOLO txt files (default: same as json_dir)
    """
    if output_dir is None:
        output_dir = json_dir
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define class mapping
    classes = ['door', 'sign']
    
    # Save classes file
    classes_file = os.path.join(Path(json_dir).parent, 'classes.txt')
    with open(classes_file, 'w') as f:
        for cls in classes:
            f.write(f"{cls}\n")
    print(f"Saved classes to: {classes_file}")
    
    # Process each JSON file
    json_files = glob.glob(os.path.join(json_dir, '*.json'))
    
    for json_path in json_files:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        img_width = data['imageWidth']
        img_height = data['imageHeight']
        
        # Output YOLO txt file
        txt_filename = Path(json_path).stem + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        
        with open(txt_path, 'w') as f:
            for shape in data['shapes']:
                label = shape['label'].lower()
                
                if label not in classes:
                    print(f"Warning: Unknown label '{label}' in {json_path}")
                    continue
                
                class_id = classes.index(label)
                points = shape['points']
                
                # Get bounding box
                x_coords = [p[0] for p in points]
                y_coords = [p[1] for p in points]
                
                x_min = min(x_coords)
                x_max = max(x_coords)
                y_min = min(y_coords)
                y_max = max(y_coords)
                
                # Convert to YOLO format (normalized center x, center y, width, height)
                x_center = ((x_min + x_max) / 2) / img_width
                y_center = ((y_min + y_max) / 2) / img_height
                width = (x_max - x_min) / img_width
                height = (y_max - y_min) / img_height
                
                # Write YOLO format line
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        print(f"Converted: {txt_filename}")
    
    print(f"\nConverted {len(json_files)} files")
    print(f"Classes: {classes}")


if __name__ == "__main__":
    # Convert train images
    convert_labelme_to_yolo("data/door_images/train")
    
    # Convert val images if they exist
    val_dir = "data/door_images/val"
    if os.path.exists(val_dir) and glob.glob(os.path.join(val_dir, '*.json')):
        convert_labelme_to_yolo(val_dir)