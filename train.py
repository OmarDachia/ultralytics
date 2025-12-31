# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n.yaml")  # build a new model from YAML
# model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

# # Train the model
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

from ultralytics import YOLO

# Load a pretrained YOLOv11n model (recommended)
model = YOLO("yolo11n.pt")

# Train the model on your custom dataset
results = model.train(
    data=r"C:\Users\Dachia\Documents\ultralytics\drone_dataset\data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="drone"
)

# Print where results are saved
print("Results saved to:", results.save_dir)