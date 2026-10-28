from ultralytics import YOLO
from pathlib import Path

data_path = Path("data/archive")



classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

for class_name in classes:
    image_path = data_path / "Train" / class_name / "images"
    labels_path = data_path / "Train" / class_name / "labels"
    

    images = list(image_path.glob("*.jpg"))
    labels = list(labels_path.glob("*.txt"))

    print("Images :", len(images))
    print("Annotations :", len(labels))


    if labels:
        label = labels[0]
        print("Fichier: ",label.name)
        print("Contenu:", label.read_text())

        for line in label.read_text().splitlines():
            values =  line.split()

            if len(values)!=5:
                print("Format incorrect :", line)
                continue

            try:
                class_id = int(values[0])
                x,y,w,h = map(float,values[1:])

                print("Class Id:", class_id)
                print("Coordonnees: ", x,y,w,h)

            except ValueError:
                print("Valeurs invalides: ",line)
