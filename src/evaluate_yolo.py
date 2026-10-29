
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    project_root = Path(__file__).resolve().parents[1]

    runs_dir = project_root / "runs" / "detect"
    checkpoints = list(
        runs_dir.glob("yolov8n_brain_tumors*/weights/best.pt")
    )
    if not checkpoints:
        raise FileNotFoundError(
            f"No trained YOLO checkpoint found under {runs_dir}. "
            "Run src/train_yolo.py first."
        )

    model_path = max(
        checkpoints,
        key=lambda checkpoint: checkpoint.parent.parent.stat().st_mtime,
    )

    data_yaml = project_root / "data" / "yolo_dataset" / "data.yaml"
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"YOLO dataset configuration not found: {data_yaml}\n"
            "Run src/prepare_dataset.py first."
        )

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Model:", model_path)
    print("Dataset:", data_yaml)
    print("Device:", device)

    model = YOLO(str(model_path))

    metrics = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=416,
        batch=16,
        device=device,
    )

    print("mAP50:", metrics.box.map50)
    print("mAP50-95:", metrics.box.map)


if __name__ == "__main__":
    main()
