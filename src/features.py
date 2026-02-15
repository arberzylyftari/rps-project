# src/features.py
import numpy as np
import tensorflow as tf


def augment_dataset(X, y, augment_factor=2, seed=42):
    """
    Augment image dataset using random flips, rotations, and zooms.

    Notes:
    - Keep augmentation light (heavy transforms can distort class signal).
    - RandomContrast removed (not needed / can hurt the model).
    - RandomZoom is zoom-out only to avoid aggressive cropping/zoom-in artifacts.

    Args:
        X: numpy array of images (N, H, W, C), values in [0, 1]
        y: numpy array of labels (N,)
        augment_factor: number of augmented copies to add per original image
        seed: random seed for reproducibility

    Returns:
        X_aug: concatenation of original + augmented images
        y_aug: corresponding labels
    """
    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, seed=seed),          # 0.15 -> 0.05
        tf.keras.layers.RandomZoom((-0.1, 0.0), seed=seed),       # zoom-out only
        tf.keras.layers.RandomBrightness(0.1, seed=seed),
        # tf.keras.layers.RandomContrast(0.1, seed=seed),         # removed
    ])

    augmented_images = [X]
    augmented_labels = [y]

    for _ in range(augment_factor):
        aug_batch = augmentation(X, training=True).numpy()
        aug_batch = np.clip(aug_batch, 0.0, 1.0)
        augmented_images.append(aug_batch)
        augmented_labels.append(y)

    X_aug = np.concatenate(augmented_images, axis=0)
    y_aug = np.concatenate(augmented_labels, axis=0)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X_aug))
    return X_aug[idx], y_aug[idx]


def extract_hog_features(X, pixels_per_cell=16, cells_per_block=2, orientations=9):
    """
    Extract HOG (Histogram of Oriented Gradients) features from images.
    """
    from skimage.feature import hog
    from skimage.color import rgb2gray

    features = []
    for img in X:
        gray = rgb2gray(img)
        hog_features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            block_norm="L2-Hys",
            feature_vector=True,
        )
        features.append(hog_features)

    return np.array(features)


def apply_standard_scaler(X_train, X_val, X_test):
    """
    Apply StandardScaler to normalize features (zero mean, unit variance).
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
