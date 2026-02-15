# src/models.py
from tensorflow import keras
import tensorflow as tf


def build_mlp_model(input_dim: int, num_classes: int):
    """
    MLP (FFNN) for flattened image vectors.
    input_dim: number of features per image after flatten
    num_classes: number of output classes
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),

        keras.layers.Dense(512),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.4),

        keras.layers.Dense(256),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"),
        keras.layers.Dropout(0.2),

        keras.layers.Dense(num_classes, activation="softmax"),
    ], name="MLP_Optimized")

    return model


def _build_cnn_augmentation(seed: int = 42):
    """
    Data augmentation applied INSIDE the CNN.
    Keras Random* layers are active ONLY during training.
    """
    return keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed),
        keras.layers.RandomRotation(0.05, seed=seed),
        keras.layers.RandomZoom((-0.1, 0.0), seed=seed),
        keras.layers.RandomBrightness(0.1, seed=seed),
    ], name="data_augmentation")


def build_cnn_model(input_shape: tuple, num_classes: int):
    """
    CNN for 2D image classification with inline data augmentation.
    Augmentation layers are active only during training.
    """
    inputs = keras.Input(shape=input_shape)

    # Augmentation (only active during training)
    x = _build_cnn_augmentation()(inputs)

    # Block 1
    x = keras.layers.Conv2D(32, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 2
    x = keras.layers.Conv2D(64, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 3
    x = keras.layers.Conv2D(128, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)
    x = keras.layers.Dropout(0.25)(x)

    # Block 4
    x = keras.layers.Conv2D(256, (3, 3), padding="same")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Classifier head
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="CNN")
    return model


def build_transfer_model(num_classes: int, input_shape: tuple = (224, 224, 3)):
    """
    Transfer learning model using MobileNetV2 pre-trained on ImageNet.
    The base model weights are frozen; only the custom head is trained.
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)

    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation("relu")(x)
    x = keras.layers.Dropout(0.5)(x)

    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="TransferLearning_MobileNetV2")
    return model
