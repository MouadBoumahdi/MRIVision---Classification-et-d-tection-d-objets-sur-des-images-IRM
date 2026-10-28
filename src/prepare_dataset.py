from pathlib import Path
import shutil

labels_path = Path("data/archive/Train/No Tumor/labels")
backup_path = Path("data/archive/Train/No Tumor/labels_backup")

if not labels_path.exists():
    raise FileNotFoundError(f"Folder not found: {labels_path.resolve()}")

# Create a backup before modifying anything
if backup_path.exists():
    raise FileExistsError(
        f"Backup already exists: {backup_path.resolve()}\n"
        "Check it before running this script again."
    )

shutil.copytree(labels_path, backup_path)

# Empty every annotation file
label_files = list(labels_path.glob("*.txt"))

for label_file in label_files:
    label_file.write_text("")

print("Backup created:", backup_path.resolve())
print("Labels emptied:", len(label_files))