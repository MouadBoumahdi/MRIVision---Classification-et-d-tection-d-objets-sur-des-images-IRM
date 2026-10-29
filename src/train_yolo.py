
import torch
from ultralytics import YOLO
from pathlib import Path


def main():
    data_yaml = Path("data/yolo_dataset/data.yaml")

    print("YAML exists:", data_yaml.exists())
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")
    print("YOLO model loaded")

    results = model.train(
        data=str(data_yaml),
        epochs=50,
        imgsz=640,
        batch=8,
        device=0,
        name="yolov8n_brain_tumors"
    )

    print("Training finished!")


if __name__ == "__main__":
    main()
