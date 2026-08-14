from ultralytics import YOLO

# Load YOLO26 small from your local archive
model_path = "D:/Sector 1/Models/YOLO/yolo26s.pt"
model = YOLO(model_path)

# Run detection on an image
results = model("path/to/your/image.jpg")

# Loop through detections and print class, confidence, and box coordinates
for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        print(f"{class_name} | confidence: {confidence:.2f} | box: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})")