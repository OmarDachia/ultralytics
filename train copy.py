from ultralytics import YOLO

# Load a pretrained YOLOv11n model (recommended)
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
results = model.train(
    data=r"C:\Users\Dachia\Documents\ultralytics\drone_dataset\data.yaml",
    epochs=50,
    imgsz=640,
    name="drone"
)

# Print where results are saved
print("Results saved to:", results.save_dir)