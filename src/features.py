# src/features.py
import numpy as np


def extract_hog_features(X, pixels_per_cell=16, cells_per_block=2, orientations=9):
    """
    Extract HOG (Histogram of Oriented Gradients) features from images.

    Reduces dimensionality from raw pixels to meaningful gradient/shape features.
    For 224x224 images with default params: ~1,764 features per image.

    Args:
        X: numpy array of images (N, H, W, C), values in [0, 1]
        pixels_per_cell: size of each HOG cell in pixels
        cells_per_block: number of cells per block for normalization
        orientations: number of gradient orientation bins

    Returns:
        features: numpy array of HOG feature vectors (N, num_hog_features)
    """
    from skimage.feature import hog
    from skimage.color import rgb2gray

    features = []
    for img in X:
        # Convert to grayscale for HOG
        gray = rgb2gray(img)
        hog_features = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=(pixels_per_cell, pixels_per_cell),
            cells_per_block=(cells_per_block, cells_per_block),
            block_norm='L2-Hys',
            feature_vector=True,
        )
        features.append(hog_features)

    return np.array(features)


def apply_standard_scaler(X_train, X_val, X_test):
    """
    Apply StandardScaler to normalize features (zero mean, unit variance).

    Critical for MLP: normalized features lead to faster convergence and better accuracy.
    Fit scaler on training data only, then transform all sets.

    Args:
        X_train: training features (2D array)
        X_val: validation features (2D array)
        X_test: test features (2D array)

    Returns:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_val_scaled, X_test_scaled, scaler
