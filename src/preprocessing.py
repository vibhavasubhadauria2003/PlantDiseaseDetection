
from pathlib import Path
import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 16
RANDOM_STATE = 42


def get_project_root():
    """Find the project root, whether called from root or notebooks."""
    current = Path.cwd()

    if current.name == "notebooks":
        current = current.parent

    return current


def load_image(image_path, label):
    """Read, resize and normalize an image to [0, 1]."""
    image = tf.io.read_file(image_path)

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32) / 255.0

    return image, label


def create_augmentation():
    """Create image augmentation layers for training data."""
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15),
        tf.keras.layers.RandomContrast(0.1)
    ])


def create_dataset(dataframe, class_names, training=False):
    """Create a TensorFlow dataset from image paths and labels."""
    class_to_index = {
        class_name: index
        for index, class_name in enumerate(class_names)
    }

    paths = dataframe["image_path"].values
    labels = dataframe["class_name"].map(class_to_index).values

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    dataset = dataset.map(
        load_image,
        num_parallel_calls=1
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=min(len(dataframe), 1000),
            seed=RANDOM_STATE
        )

        augmentation = create_augmentation()

        dataset = dataset.map(
            lambda image, label: (
                augmentation(image, training=True),
                label
            ),
            num_parallel_calls=1
        )

    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(1)

    return dataset
