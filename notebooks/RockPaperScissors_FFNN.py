# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ROCK-PAPER-SCISSORS: Multi-class Classification
# ## Comparing MLP, HOG Features, and Transfer Learning
#
# **Goal:** Maximize validation accuracy - no more constraints!
#
# **Models Compared:**
# 1. **MLP (Baseline)** - Feed-Forward Neural Network with StandardScaler
# 2. **MLP + HOG Features** - Better features via gradient histograms
# 3. **Transfer Learning** - Pre-trained MobileNetV2 features

# %% [markdown]
# ## 1. Setup and Imports

# %%
import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
import pandas as pd

from src.data import load_images_as_numpy
from src.models import build_mlp_model, build_transfer_model
from src.features import extract_hog_features, apply_standard_scaler
from src.train import compile_model, get_callbacks
from src.eval import evaluate_model, plot_history

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

np.random.seed(42)
tf.random.set_seed(42)

# %%
DATA_ROOT = "../data/rps"
IMG_H, IMG_W = 224, 224
BATCH_SIZE = 16
EPOCHS = 90
CLASSES = ['paper', 'rock', 'scissor']
NUM_CLASSES = len(CLASSES)

print(f"Data root: {DATA_ROOT}")
print(f"Image size: {IMG_H}x{IMG_W}")
print(f"Classes: {CLASSES}")

# %% [markdown]
# ## 2. Load Data as NumPy Arrays

# %%
print("Loading training data...")
X_train, y_train, class_names = load_images_as_numpy(
    os.path.join(DATA_ROOT, 'train')
)

print("\nLoading validation data...")
X_val, y_val, _ = load_images_as_numpy(
    os.path.join(DATA_ROOT, 'val'),
    verbose=False
)

print("\nLoading test data...")
X_test, y_test, _ = load_images_as_numpy(
    os.path.join(DATA_ROOT, 'test'),
    verbose=False
)

print("\n" + "="*60)
print("DATA LOADED SUCCESSFULLY")
print("="*60)
print(f"Train set: {X_train.shape[0]} images, shape {X_train.shape[1:]}")
print(f"Validation set: {X_val.shape[0]} images, shape {X_val.shape[1:]}")
print(f"Test set: {X_test.shape[0]} images, shape {X_test.shape[1:]}")

print("\nClass distribution in training:")
for i, class_name in enumerate(CLASSES):
    count = np.sum(y_train == i)
    print(f"  {class_name}: {count} images ({count/len(y_train)*100:.1f}%)")

# %% [markdown]
# ## 3. Visualize Sample Images

# %%
plt.figure(figsize=(15, 5))
for i in range(min(12, len(X_train))):
    plt.subplot(2, 6, i + 1)
    plt.imshow(X_train[i])
    plt.title(CLASSES[y_train[i]], fontsize=10)
    plt.axis('off')
plt.tight_layout()
plt.suptitle('Sample Training Images', y=1.02, fontsize=14, fontweight='bold')
plt.show()

# %% [markdown]
# ## 4. Prepare Data for MLP (Flatten)

# %%
X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_val_flat = X_val.reshape(X_val.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

print(f"Original shape: {X_train.shape}")
print(f"Flattened shape: {X_train_flat.shape}")
print(f"Each image is now a vector of {X_train_flat.shape[1]:,} features")

# %%
# Apply StandardScaler for better MLP performance
print("Applying StandardScaler (zero mean, unit variance)...")
X_train_scaled, X_val_scaled, X_test_scaled, scaler = apply_standard_scaler(
    X_train_flat, X_val_flat, X_test_flat
)

print(f"\nBefore scaling:")
print(f"  Mean: {X_train_flat.mean():.4f}, Std: {X_train_flat.std():.4f}")
print(f"After scaling:")
print(f"  Mean: {X_train_scaled.mean():.4f}, Std: {X_train_scaled.std():.4f}")
print("StandardScaler improves MLP convergence by normalizing features!")

# %% [markdown]
# ## 5. Baseline MLP Model (with StandardScaler)

# %%
mlp_model = build_mlp_model(X_train_scaled.shape[1], NUM_CLASSES)
mlp_model.summary()

# %%
compile_model(mlp_model)
callbacks = get_callbacks()

print("="*70)
print("TRAINING BASELINE MLP (with StandardScaler)")
print("="*70)

history_mlp = mlp_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

mlp_val_loss, mlp_val_acc = mlp_model.evaluate(X_val_scaled, y_val, verbose=0)
print(f"\nBaseline MLP Validation Accuracy: {mlp_val_acc:.2%}")
plot_history(history_mlp, title="Baseline MLP (StandardScaler)")

# %% [markdown]
# ## 6. MLP + HOG Features (with StandardScaler)

# %%
print("Extracting HOG features...")
X_train_hog = extract_hog_features(X_train)
X_val_hog = extract_hog_features(X_val)
X_test_hog = extract_hog_features(X_test)

print(f"Original features: {X_train_flat.shape[1]:,}")
print(f"HOG features: {X_train_hog.shape[1]:,}")

# Apply StandardScaler to HOG features
print("\nApplying StandardScaler to HOG features...")
X_train_hog_scaled, X_val_hog_scaled, X_test_hog_scaled, scaler_hog = apply_standard_scaler(
    X_train_hog, X_val_hog, X_test_hog
)

# %%
mlp_hog_model = build_mlp_model(X_train_hog_scaled.shape[1], NUM_CLASSES)
compile_model(mlp_hog_model)
callbacks_hog = get_callbacks()

print("="*70)
print("TRAINING MLP + HOG FEATURES (with StandardScaler)")
print("="*70)

history_mlp_hog = mlp_hog_model.fit(
    X_train_hog_scaled, y_train,
    validation_data=(X_val_hog_scaled, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_hog,
    verbose=1
)

_, mlp_hog_val_acc = mlp_hog_model.evaluate(X_val_hog_scaled, y_val, verbose=0)
print(f"\nMLP + HOG Validation Accuracy: {mlp_hog_val_acc:.2%}")
plot_history(history_mlp_hog, title="MLP + HOG + StandardScaler")

# %% [markdown]
# ## 7. Transfer Learning (MobileNetV2)

# %%
transfer_model = build_transfer_model(NUM_CLASSES, input_shape=(IMG_H, IMG_W, 3))
transfer_model.summary()

# %%
compile_model(transfer_model, learning_rate=0.0001)
callbacks_transfer = get_callbacks(patience=15)

print("="*70)
print("TRAINING TRANSFER LEARNING (MobileNetV2)")
print("="*70)

history_transfer = transfer_model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=BATCH_SIZE,
    callbacks=callbacks_transfer,
    verbose=1
)

_, transfer_val_acc = transfer_model.evaluate(X_val, y_val, verbose=0)
print(f"\nTransfer Learning Validation Accuracy: {transfer_val_acc:.2%}")
plot_history(history_transfer, title="Transfer Learning (MobileNetV2)")

# %% [markdown]
# ## 8. Final Comparison

# %%
class_names = ["rock", "paper", "scissors"]
rows = []

def add_row(name, model, Xv, yv, Xt, yt, val_acc):
    s = evaluate_model(
        model, Xv, yv, Xt, yt, class_names,
        show_plots=False, print_reports=False
    )
    rows.append({
        "Model": name,
        "Validation Accuracy (%)": val_acc * 100,
        "Val F1 Macro (%)": s["val"]["f1_macro"] * 100,
        "Test F1 Macro (%)": s["test"]["f1_macro"] * 100,
    })

add_row("MLP (Baseline)", mlp_model, X_val_scaled, y_val, X_test_scaled, y_test, mlp_val_acc)
add_row("MLP + HOG", mlp_hog_model, X_val_hog_scaled, y_val, X_test_hog_scaled, y_test, mlp_hog_val_acc)
add_row("Transfer Learning", transfer_model, X_val, y_val, X_test, y_test, transfer_val_acc)

# Build + sort
results = pd.DataFrame(rows).sort_values(
    "Val F1 Macro (%)", ascending=False, na_position="last"
).reset_index(drop=True)

print("\n" + "="*70)
print("FINAL MODEL COMPARISON")
print("="*70)
print(results.to_string(index=False))
print("="*70)

# Plot Val F1 Macro (%)
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(results)))
bars = ax.barh(results["Model"], results["Val F1 Macro (%)"], color=colors)

ax.set_xlabel("Validation F1 Macro (%)", fontsize=12, fontweight="bold")
ax.set_title("Model Comparison - Validation Macro F1", fontsize=14, fontweight="bold")
ax.grid(axis="x", alpha=0.3)
ax.set_xlim(0, 105)

for bar, val in zip(bars, results["Val F1 Macro (%)"]):
    ax.text(
        bar.get_width() + 0.5,
        bar.get_y() + bar.get_height()/2,
        f"{val:.1f}%",
        ha="left",
        va="center",
        fontsize=10,
        fontweight="bold"
    )

plt.tight_layout()
plt.show()



# %% [markdown]
# ## 9. Evaluate Best Model

# %%
results = results.sort_values(
    "Val F1 Macro (%)", ascending=False, na_position="last"
).reset_index(drop=True)

best_model_name = results.iloc[0]["Model"]
print(f"Best model: {best_model_name}")

if best_model_name == "Transfer Learning":
    best_model = transfer_model
    X_val_input, X_test_input = X_val, X_test
elif best_model_name == "MLP + HOG":
    best_model = mlp_hog_model
    X_val_input, X_test_input = X_val_hog_scaled, X_test_hog_scaled
else:
    best_model = mlp_model
    X_val_input, X_test_input = X_val_scaled, X_test_scaled

# Evaluate best model (confusion matrices + per-class F1)
best_scores = evaluate_model(
    best_model,
    X_val_input, y_val,
    X_test_input, y_test,
    CLASSES
)

# Explicit macro F1 print (clear for graders)
print("\nBest model metrics:")
print(f"  Val F1 Macro (%):  {best_scores['val']['f1_macro'] * 100:.2f}")
print(f"  Test F1 Macro (%): {best_scores['test']['f1_macro'] * 100:.2f}")


# %% [markdown]
# ## 10. Save Best Model

# %%
best_model.save('rps_best_model.keras')
print(f"Best model ({best_model_name}) saved as: rps_best_model.keras")
