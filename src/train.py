# src/train.py
from tensorflow import keras


def compile_model(model, learning_rate: float = 0.001):
    """Compile model with Adam optimizer and sparse categorical crossentropy."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )


def get_callbacks(patience: int = 10, min_lr: float = 1e-6):
    """
    Return standard training callbacks:
    - EarlyStopping: stops when val_accuracy stops improving
    - ReduceLROnPlateau: halves learning rate when val_loss plateaus
    """
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=patience,
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=min_lr,
        verbose=1,
    )

    return [early_stopping, reduce_lr]
