from ultralytics import YOLO
import glob

model = YOLO('models/door_detector.pt')

test_image = glob.glob('data/door_images/train/*.jpg')[0]

results = model(test_image, save=True)

print(f"Tested on: {test_image}")
print(f"Results saved to: runs/detect/predict/")

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        name = model.names[cls]
        print(f"Detected: {name} ({conf:.2f})")