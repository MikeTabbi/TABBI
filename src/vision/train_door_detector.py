from ultralytics import YOLO

def train():
    model = YOLO('yolov8n.pt')
    
    results = model.train(
        data='data/door_images/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=8,
        name='door_detector'
    )
    
    print("Training complete!")
    print(f"Model saved to: runs/detect/door_detector/weights/best.pt")

if __name__ == "__main__":
    train()