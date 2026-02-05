# src/data.py
import os
import numpy as np
import tensorflow as tf

def load_images_as_numpy(directory, img_size=(224, 224), verbose=True):
    """Load all images from directory structure into numpy arrays.
    Expects: directory/class_name/*.png|jpg|jpeg
    Returns: (X, y, class_names)
    """
    images = []
    labels = []

    class_names = sorted(os.listdir(directory))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue

        if verbose:
            print(f"Loading {class_name}...", end=" ")
        count = 0

        for img_name in os.listdir(class_dir):
            if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = tf.keras.preprocessing.image.load_img(
                        img_path, target_size=img_size
                    )
                    img_array = tf.keras.preprocessing.image.img_to_array(img)
                    img_array = img_array / 255.0

                    images.append(img_array)
                    labels.append(class_to_idx[class_name])
                    count += 1
                except Exception as e:
                    if verbose:
                        print(f"\nError loading {img_path}: {e}")

        if verbose:
            print(f"{count} images")

    X = np.array(images)
    y = np.array(labels)
    return X, y, class_names
