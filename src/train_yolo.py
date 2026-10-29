
import torch
from ultralytics import YOLO
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_yaml = project_root / "data" / "yolo_dataset" / "data.yaml"

    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"YOLO dataset configuration not found: {data_yaml}\n"
            "Run src/prepare_dataset.py first."
        )

    print("Dataset YAML:", data_yaml)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov8n.pt")
    print("YOLO model loaded")

    results = model.train(
        data=str(data_yaml),
        epochs=15,
        imgsz=416,
        batch=16,
        device=0 if torch.cuda.is_available() else "cpu",
        workers=4,
        cache="ram",
        name="yolov8n_brain_tumors"
    )

    print("Training finished!")


if __name__ == "__main__":
    main()
