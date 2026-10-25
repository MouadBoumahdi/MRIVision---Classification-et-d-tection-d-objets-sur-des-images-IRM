
from pathlib import Path
import tensorflow as tf

# 1. Configuration
DATA_PATH = Path("data/archive")  # À adapter selon ton dossier
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

class_names = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# 2. Data augmentation (Train uniquement)
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomTranslation(0.05, 0.05),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# 3. Charger et préparer une image
def load_image(path, label, training=False):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)

    if training:
        image = augmentation(image, training=True)
        image = tf.clip_by_value(image, 0, 255)

    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

    return image, label

# 4. Créer un dataset
def create_dataset(split, training=False):
    paths = []
    labels = []

    for i, name in enumerate(class_names):
        folder = DATA_PATH / split / name / "images"

        for path in sorted(folder.glob("*.jpg")):
            paths.append(str(path))
            labels.append(i)

    if not paths:
        raise ValueError(f"Aucune image trouvée dans {split}")

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(len(paths))

    ds = ds.map(
        lambda path, label: load_image(path, label, training),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    print(f"{split}: {len(paths)} images")

    return ds

# 5. Créer les datasets
train_ds = create_dataset("Train", training=True)
val_ds = create_dataset("Val")
test_ds = create_dataset("Test")

# 6. Vérifier les images
for images, labels in test_ds.take(1):
    print("Shape:", images.shape)
    print("Pixel min:", tf.reduce_min(images).numpy())
    print("Pixel max:", tf.reduce_max(images).numpy())
    print("Labels:", labels[:10].numpy())
