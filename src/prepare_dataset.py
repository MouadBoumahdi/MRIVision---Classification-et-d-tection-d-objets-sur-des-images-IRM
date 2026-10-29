
from pathlib import Path
import shutil

project_root = Path(__file__).resolve().parents[1]
source_root = project_root / "data" / "archive"
target_root = project_root / "data" / "yolo_dataset"

splits = {
    "Train": "train",
    "Val": "val",
    "Test": "test",
}

classes = {
    "Glioma": 0,
    "Meningioma": 1,
    "No Tumor": None,
    "Pituitary": 2,
}

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for source_split, target_split in splits.items():
    for class_name, class_id in classes.items():

        source_dir = source_root / source_split / class_name
        images_dir = source_dir / "images"
        labels_dir = source_dir / "labels"

        target_images = target_root / "images" / target_split
        target_labels = target_root / "labels" / target_split

        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(
                f"Missing images or labels folder: {source_dir}"
            )

        target_images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)

        for image_path in images_dir.iterdir():
            if (
                not image_path.is_file()
                or image_path.suffix.lower() not in image_extensions
            ):
                continue

            label_path = labels_dir / f"{image_path.stem}.txt"

            if class_id is not None and not label_path.exists():
                raise FileNotFoundError(
                    f"Missing label for image: {image_path}"
                )

            # Copy image without changing the original
            shutil.copy2(
                image_path,
                target_images / image_path.name
            )

            # Prepare the label for the 3-class detector
            destination_label = target_labels / label_path.name

            if class_id is None:
                # No Tumor: negative image, no bounding boxes
                destination_label.write_text("", encoding="utf-8")
            else:
                updated_lines = []

                for line in label_path.read_text(
                    encoding="utf-8"
                ).splitlines():
                    parts = line.split()

                    if not parts:
                        continue

                    parts[0] = str(class_id)
                    updated_lines.append(" ".join(parts))

                destination_label.write_text(
                    "\n".join(updated_lines)
                    + ("\n" if updated_lines else ""),
                    encoding="utf-8"
                )

print("Images and labels copied successfully!")
