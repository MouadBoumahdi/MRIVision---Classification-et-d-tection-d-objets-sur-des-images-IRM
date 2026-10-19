from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt


data_path = Path("data/archive")
images = list(data_path.rglob("*.jpg"))

print("Image count:", len(images))

print("Classes Name:")
for item in (data_path/"Train").iterdir():
    print(item.name)


for image in images[:5]:
    img = Image.open(image)

    print("Image:", image.name)
    print("Size:", img.size)
    print("Mode:", img.mode)
    print("Format:", img.format)





# corrupted images
corrupted_images= []

for image in images:
    try:
        with Image.open(image) as img:
            img.verify()
    except Exception:
        corrupted_images.append(image)

print("Corrupted images:", len(corrupted_images))
for image in corrupted_images:
    print(image)



for image in images[:5]:
    img = Image.open(image)

    plt.imshow(img, cmap="gray")
    plt.title(image.parent.parent.name)
    plt.axis("off")
    plt.show()


    img.close()






# annotation:information added to image to
# help the model undersatand what is in the
# image or where its located(labels)

train_labels = list(data_path.glob("Train/*/labels/*.txt"))
val_labels = list(data_path.glob("Val/*/labels/*.txt"))

print("Train annotations:", len(train_labels))
print("Validation annotations:", len(val_labels))



invalid = 0 
for label in data_path.rglob("*.txt"):
    for line in label.read_text().splitlines():
        values = line.split()

        if len(values) != 5:
            invalid +=1

print("Invalid annotationP:",invalid)









