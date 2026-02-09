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


def build_cnn_model(input_shape: tuple, num_classes: int):
    """
    CNN for 2D image classification. Learns spatial features via convolutions.
    input_shape: (H, W, C) e.g. (224, 224, 3)
    num_classes: number of output classes
    """
    model = keras.Sequential([
        keras.layers.Input(shape=input_shape),

        # Block 1
        keras.layers.Conv2D(32, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Block 2
        keras.layers.Conv2D(64, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Block 3
        keras.layers.Conv2D(128, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Dropout(0.25),

        # Block 4
        keras.layers.Conv2D(256, (3, 3), padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.GlobalAveragePooling2D(),

        # Classifier head
        keras.layers.Dense(256),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(num_classes, activation='softmax'),
    ], name="CNN")

    return model


def build_transfer_model(num_classes: int, input_shape: tuple = (224, 224, 3)):
    """
    Transfer learning model using MobileNetV2 pre-trained on ImageNet.
    The base model weights are frozen; only the custom head is trained.
    num_classes: number of output classes
    input_shape: (H, W, C) - must be at least (96, 96, 3) for MobileNetV2
    """
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
    )
    base_model.trainable = False

    inputs = keras.Input(shape=input_shape)
    # MobileNetV2 expects inputs in [-1, 1]; preprocess accordingly
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs, outputs, name="TransferLearning_MobileNetV2")
    return model
